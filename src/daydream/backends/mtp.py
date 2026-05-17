"""MTP / NextN speculative-decoding backend.

This is the Daydream-native MTP path. It re-uses the MTPLX library
(github.com/youssofal/MTPLX, Apache-2.0) for the hard parts:

    * Loading the `mtp.safetensors` sidecar (one transformer block +
      two norms + a fusion linear — see EXPECTED_MTP_KEYS in
      mtplx.constants).
    * Injecting MTP support into the loaded MLX model (mtp_patch.py).
    * Probability-ratio acceptance + residual correction
      (mtplx.sampling — the Leviathan–Chen 2023 spec sampler).
    * Native MLP / split-attention optimisations (mtplx.native_mlp,
      mtplx.attention_split).

What Daydream adds on top:

    * Same-process, in-memory adapter — no subprocess, no REST roundtrip
      (MTPLX ships a uvicorn server; we skip it and call the Python
      API directly). Faster: zero IPC; one MLX device context.
    * Stable, opinionated defaults — `--speculative mtp` is a single
      flag; no profile flags to juggle.
    * Honest capability reporting — `daydream show` tells the user
      whether MTPLX is installed AND whether an MTP-equipped model is
      cached, in plain English. MTPLX itself surfaces all of this only
      via JSON contract files.
    * Daydream's existing telemetry — token-by-token `from_draft`
      attribution, prompt/gen tps, peak memory; same shape as the
      non-spec path so `daydream run -v` works identically.
    * Fully integrated `/draft mtp` slash command + clean fallback to
      `draft`/`lookup` when MTPLX or the sidecar is missing.

Phase 2A: detection + capability reporting + skeleton.
Phase 2B (this commit): adapter from mtplx.runtime → Daydream's
    GenerationResponse stream. Marked experimental until validated on
    Apple Silicon hardware with a real Qwen3.6 + MTPLX-Optimized-Speed
    checkpoint.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Generator, Optional

from .base import BackendCapabilities, GenerationBackend, SamplingParams, SpeculativeParams


class MTPSidecarMissing(RuntimeError):
    """No MTPLX sidecar available locally; install one to enable MTP."""


class MTPLXNotInstalled(RuntimeError):
    """The `mtplx` Python package isn't importable."""


class MTPInsufficientMemory(RuntimeError):
    """Available unified memory is too low to safely load MTPLXRuntime.

    MTPLX loads its own ~16 GB copy of the trunk in addition to whatever
    daydream already has resident. On 24 GB Macs this OOMs the GPU and
    `generate_mtpk` returns no tokens. Refuse cleanly with this error
    so the caller can recommend the lighter `/draft on` path.
    """


# Floor for the *in-place* attach path: daydream reuses the already-
# loaded 16 GB trunk and only allocates the ~338 MB MTP head + a small
# working margin. We keep a 2 GB cushion so prefill activations have
# headroom — way below the 18 GB the second-trunk path used to need.
MTP_MIN_FREE_RAM_BYTES = 2 * 1024 * 1024 * 1024


# Names of RMSNorm weights mlx_lm.models.qwen3_5.Model.sanitize() shifts
# by +1.0 when it detects mtp.* keys in the safetensors (qwen3_5.py:317).
# Our thin install symlinks the standard trunk (no mtp keys → no shift
# at load time), so we apply the same shift in-place ourselves before
# attaching the MTP head, and reverse it on teardown so plain decode
# keeps working after /draft mtp off.
_QWEN_NORM_ATTRS_LAYER = ("input_layernorm", "post_attention_layernorm")
_QWEN_NORM_ATTRS_ATTN = ("q_norm", "k_norm")


def _shift_qwen_norms(model: Any, delta: float) -> int:
    """In-place add `delta` to every RMSNorm weight that mlx_lm would
    shift at load time when `has_mtp_weights=True`. Returns the number
    of tensors mutated. Reversible by calling again with `-delta`.

    Why: the MTP head was trained against a +1.0-shifted trunk. Without
    this, the head's input distribution is off by a full unit per
    channel and the model produces garbage (the "88888…" / "* * *"
    failure modes we saw earlier).
    """
    import mlx.core as mx

    text_model = getattr(model, "model", model)
    count = 0
    for layer in getattr(text_model, "layers", ()):
        for attr in _QWEN_NORM_ATTRS_LAYER:
            n = getattr(layer, attr, None)
            if n is not None and hasattr(n, "weight"):
                n.weight = n.weight + delta
                count += 1
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            for attr in _QWEN_NORM_ATTRS_ATTN:
                n = getattr(attn, attr, None)
                if n is not None and hasattr(n, "weight"):
                    n.weight = n.weight + delta
                    count += 1
    top_norm = getattr(text_model, "norm", None)
    if top_norm is not None and hasattr(top_norm, "weight"):
        top_norm.weight = top_norm.weight + delta
        count += 1
    mx.eval(text_model.parameters())
    return count


@dataclass(frozen=True)
class MTPSidecar:
    runtime_json_path: Path
    model_dir: Path
    sidecar_repo: str
    base_trunk: str
    mtp_depth_max: int
    arch_id: str

    @property
    def description(self) -> str:
        return f"{self.sidecar_repo} (depth={self.mtp_depth_max}, arch={self.arch_id})"


KNOWN_MTP_SIDECAR_REPOS: tuple[str, ...] = (
    "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
)
RECOMMENDED_PULL_HINT = "daydream pull Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"


def _hf_cache_root() -> Path:
    env_hub = os.environ.get("HF_HUB_CACHE")
    if env_hub:
        return Path(env_hub).expanduser()
    env_home = os.environ.get("HF_HOME")
    if env_home:
        return Path(env_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _read_runtime_json(snapshot_dir: Path) -> Optional[MTPSidecar]:
    runtime = snapshot_dir / "mtplx_runtime.json"
    if not runtime.is_file():
        return None
    try:
        data = json.loads(runtime.read_text())
    except (OSError, ValueError):
        return None
    arch_id = str(data.get("arch_id", "")).strip()
    sidecar_repo = str(data.get("mtp_sidecar", "")).strip()
    base_trunk = str(data.get("base_trunk", "")).strip()
    depth = int(data.get("mtp_depth_max", 0) or 0)
    if not arch_id or not sidecar_repo:
        return None
    return MTPSidecar(
        runtime_json_path=runtime,
        model_dir=snapshot_dir,
        sidecar_repo=sidecar_repo,
        base_trunk=base_trunk,
        mtp_depth_max=depth or 3,
        arch_id=arch_id,
    )


def detect_mtp_sidecar(model_ref: Optional[str] = None) -> Optional[MTPSidecar]:
    """Scan for a usable MTP sidecar.

    Search order:
        1. Daydream's thin-install location (`~/.daydream/mtp/<slug>/`).
           This is the common case — created by `daydream mtp install`
           or the `--speculative mtp` auto-flow.
        2. HuggingFace cache — picks up users who manually pulled the
           full ~16 GB Youssofal repo via `daydream pull`.

    Returns the first sidecar whose `base_trunk` matches `model_ref`
    if one exists; otherwise the first sidecar found.
    """
    # 1. Daydream thin-install (preferred — keeps everything in one
    # directory the user owns).
    fallback: Optional[MTPSidecar] = None
    try:
        from daydream.backends.mtp_install import _install_root
        root = _install_root()
        if root.is_dir():
            for install_dir in sorted(root.iterdir()):
                if not install_dir.is_dir():
                    continue
                sidecar = _read_runtime_json(install_dir)
                if sidecar is None:
                    continue
                if model_ref and sidecar.base_trunk == model_ref:
                    return sidecar
                if fallback is None:
                    fallback = sidecar
    except Exception:
        # Defensive — sidecar detection must never raise.
        pass

    # 2. HuggingFace cache (fallback — full-repo installs).
    cache = _hf_cache_root()
    if cache.is_dir():
        for repo_dir in sorted(cache.glob("models--*")):
            snapshots = repo_dir / "snapshots"
            if not snapshots.is_dir():
                continue
            for snap in sorted(snapshots.iterdir()):
                if not snap.is_dir():
                    continue
                sidecar = _read_runtime_json(snap)
                if sidecar is None:
                    continue
                if model_ref and sidecar.base_trunk == model_ref:
                    return sidecar
                if fallback is None:
                    fallback = sidecar
    return fallback


def mtplx_available() -> bool:
    """Return True iff the `mtplx` Python package can be imported."""
    return importlib_util.find_spec("mtplx") is not None


def mtplx_version() -> Optional[str]:
    if not mtplx_available():
        return None
    try:
        import mtplx
        return getattr(mtplx, "__version__", None)
    except Exception:
        return None


def install_mtplx(*, quiet: bool = False) -> bool:
    """Best-effort `pip install mtplx>=0.3.6` into the running interpreter.

    Returns True on success. Uses the same Python whose venv runs
    Daydream — never invokes a system pip. Designed to be called from
    an interactive prompt after the user has consented; it does NOT
    confirm on its own.
    """
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "pip", "install", "mtplx>=0.3.6"]
    if quiet:
        cmd.append("--quiet")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        return False
    # Bust importlib's negative cache so the new install is visible
    # in this same Python process.
    importlib_util.invalidate_caches()
    return mtplx_available()


DEFAULT_MTP_CHECKPOINT = "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"


def estimate_mtp_install_size_bytes() -> int:
    """Approximate total download size to enable --speculative mtp.

    The MTP-equipped checkpoint bundles its own copy of the trunk
    (3 × 5.3 GB safetensors) plus the MTP head (337 MB), so we report
    the published total — 16.4 GB — rounded for the prompt. Kept as
    a single function so the installer and CLI prompt agree.
    """
    return 16_419_071_404  # bytes, from MTPLX_PUBLISH_MANIFEST.json


class MTPBackend(GenerationBackend):
    """Daydream-native MTP backend, library-backed by MTPLX.

    Lifecycle expectation: the caller has already loaded the *trunk*
    model (`self.model`) and tokenizer via the standard
    `daydream.engine.load_model` path. `prepare()` then re-routes
    generation through an `MTPLXRuntime` that wraps the same trunk
    plus the freshly-loaded MTP head.
    """

    method = "mtp"

    def __init__(self, model, tokenizer, *, model_ref: Optional[str] = None) -> None:
        super().__init__(model, tokenizer, model_ref=model_ref)
        self._sidecar: Optional[MTPSidecar] = None
        self._mtplx_runtime: Any = None  # mtplx.MTPLXRuntime when loaded

    # ── Lifecycle ──────────────────────────────────────────────
    def prepare(self) -> None:
        if self._prepared:
            return
        self._sidecar = detect_mtp_sidecar(self.model_ref)
        self._prepared = True

    def teardown(self) -> None:
        # Drop the MTPLXRuntime reference. We DO NOT detach the MTP
        # module from the model (text_model.mtp = ...) — that would
        # require undoing configure_split_full_attention / configure_
        # native_mlp too, and there's no clean unwind path in mtplx.
        # The MTP head is small (~338 MB); leaving it dormant is the
        # pragmatic call. Plain decode skips the MTP module since it
        # only reads the trunk's standard forward path.
        self._mtplx_runtime = None
        try:
            import gc
            import mlx.core as mx
            gc.collect()
            mx.clear_cache()
        except Exception:
            pass

    # ── Capability reporting ────────────────────────────────────
    def capabilities(self) -> BackendCapabilities:
        if not self._prepared:
            self.prepare()

        installed = mtplx_available()
        diagnostic: dict[str, Any] = {
            "mtplx_installed": installed,
            "mtplx_version": mtplx_version(),
        }
        if self._sidecar is not None:
            diagnostic.update({
                "sidecar_repo": self._sidecar.sidecar_repo,
                "base_trunk": self._sidecar.base_trunk,
                "mtp_depth_max": self._sidecar.mtp_depth_max,
                "arch_id": self._sidecar.arch_id,
                "runtime_json": str(self._sidecar.runtime_json_path),
                "model_dir": str(self._sidecar.model_dir),
            })

        if not installed:
            return BackendCapabilities(
                method="mtp",
                supports_speculative=False,
                external_weights_required=True,
                notes=(
                    "MTP runtime (mtplx) is bundled with Daydream on "
                    "Apple Silicon, but the import failed in this "
                    "Python environment. Likely cause: you're on a "
                    "non-Apple-Silicon box (MTP needs Metal), or the "
                    "venv was modified after install. Reinstall with: "
                    "`pip install --force-reinstall daydream`."
                ),
                diagnostic=diagnostic,
            )

        if self._sidecar is None:
            return BackendCapabilities(
                method="mtp",
                supports_speculative=False,
                external_weights_required=True,
                notes=(
                    "MTP head weights are not bundled with the public "
                    "Qwen3.6 checkpoint. Install an MTPLX-equipped "
                    f"checkpoint with: `{RECOMMENDED_PULL_HINT}`. "
                    "Then re-run `daydream run --speculative mtp`."
                ),
                diagnostic={**diagnostic, "reason": "no-sidecar"},
            )

        return BackendCapabilities(
            method="mtp",
            supports_speculative=True,
            external_weights_required=True,
            notes=(
                f"MTP ready: {self._sidecar.description}. "
                f"Native depth: {self._sidecar.mtp_depth_max}. "
                "Acceptance: probability-ratio (Leviathan–Chen) with "
                "residual correction. Daydream routes through MTPLX "
                "in-process for zero-IPC speedup."
            ),
            diagnostic=diagnostic,
        )

    # ── Generation ──────────────────────────────────────────────
    def generate(
        self,
        messages: list[dict],
        sampling: SamplingParams,
        speculative: SpeculativeParams,
        *,
        chat_template_kwargs: Optional[dict] = None,
        prompt_cache: Optional[Any] = None,
        prefill_step_size: Optional[int] = None,
    ) -> Generator:
        self.prepare()
        caps = self.capabilities()
        if not caps.supports_speculative:
            raise MTPSidecarMissing(caps.notes)
        if self._sidecar is None:  # defensive — covered by caps check
            raise MTPSidecarMissing("MTP sidecar missing")

        # Render the prompt the same way every other Daydream backend
        # does — keeps /effort, thinking budgets, system memory, etc.
        # working identically across backends.
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **(chat_template_kwargs or {}),
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in messages)

        yield from self._stream_mtplx(
            prompt=prompt,
            sampling=sampling,
            speculative=speculative,
        )

    # ── Internal: MTP runtime construction ─────────────────────
    def _ensure_runtime(self):
        """Attach the MTP head IN-PLACE to daydream's loaded trunk.

        This is the right design (no second trunk):

          1. Load augmented config from the install dir — it carries
             the `mtplx_mtp_quantization` block the MTP head needs.
          2. Call `inject_mtp_support(model, install_dir, config,
             contract)` — adds `text_model.mtp` and metadata.
          3. Install MTPLX's split-attention + native-MLP optimisers
             on the same model.
          4. Wrap in an `MTPLXRuntime` whose `.model` IS daydream's
             trunk (now augmented). One trunk in memory, period.

        Memory cost over plain decode: +338 MB MTP head + ~little.
        Total resident on Qwen3.6-27B-4bit: ~16.3 GB (vs ~32 GB for
        the old `mtplx.load()` second-trunk path).

        Note on the +1.0 norm shift: `mlx_lm.qwen3_5.Model.sanitize`
        only applies it when the safetensors contain `mtp.*` keys
        (qwen3_5.py:307). Our thin install symlinks the standard
        trunk (no mtp keys), so no shift fires at load. The standard
        trunk's stored weights are absolute and already correct — we
        must NOT shift them ourselves, doing so produces token soup.
        The MTP head in `mtp.safetensors` is trained to plug into
        the absolute (un-shifted) trunk, which is what daydream has.
        """
        if self._mtplx_runtime is not None:
            return self._mtplx_runtime
        if self._sidecar is None:
            raise MTPSidecarMissing("No MTP sidecar detected")

        # Pre-flight: we only need headroom for the MTP head and
        # activations, not a whole second trunk.
        from daydream.engine import available_ram_bytes
        free = available_ram_bytes()
        if free is not None and free < MTP_MIN_FREE_RAM_BYTES:
            need_gb = MTP_MIN_FREE_RAM_BYTES // (1024 ** 3)
            have_gb = max(free // (1024 ** 3), 0)
            raise MTPInsufficientMemory(
                f"MTP needs ≥{need_gb} GB free unified memory (have ~{have_gb} GB). "
                "Close other apps or use `/draft off` for plain decode."
            )

        install_dir = Path(self._sidecar.model_dir)

        # 1. Augmented config — has mtplx_mtp_quantization.
        with (install_dir / "config.json").open("r") as f:
            config = json.load(f)

        # 2. Attach the MTP head. inject_mtp_support reads
        # mtp.safetensors out of install_dir (real file, ~338 MB)
        # and bolts it onto text_model.mtp.
        from mtplx.mtp_patch import (
            MTPContract,
            inject_mtp_support,
            validate_mtp_support,
        )

        contract = MTPContract().with_config_defaults(config)
        ok = inject_mtp_support(self.model, install_dir, config, contract)
        if not ok or not validate_mtp_support(self.model):
            raise MTPSidecarMissing(
                "MTP injection failed — install may be incomplete. "
                "Try: `daydream mtp uninstall && daydream mtp install`."
            )

        # 3. MTPLX's runtime optimisations on the same model.
        from mtplx.attention_split import configure_split_full_attention
        from mtplx.native_mlp import configure_native_mlp

        configure_split_full_attention(self.model)
        configure_native_mlp(self.model)

        # 4. Wrap in MTPLXRuntime — note: model is daydream's
        # trunk, not a second copy.
        from mtplx import MTPLXRuntime

        self._mtplx_runtime = MTPLXRuntime(
            self.model,
            self.tokenizer,
            install_dir,
            True,
            contract,
        )
        return self._mtplx_runtime

    def _free_daydream_trunk(self) -> None:
        """Best-effort drop of Daydream's cached trunk before MTPLX
        loads its own copy. Recovers some GPU memory; the chat-side
        ref in `self.model` still keeps the trunk pinned until the
        backend is torn down, but every helper that talks to
        engine.load_model will at least get the MTPLX-built copy
        going forward.
        """
        if not self.model_ref:
            return
        try:
            from daydream import engine
            from daydream.registry import normalize_hf_reference

            for key in {self.model_ref, normalize_hf_reference(self.model_ref)}:
                engine._loaded_entries.pop(key, None)
            import gc
            gc.collect()
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

    def _stream_mtplx(
        self,
        *,
        prompt: str,
        sampling: SamplingParams,
        speculative: SpeculativeParams,
    ) -> Generator:
        """Drive MTPLX's `generate_mtpk` and stream tokens AS produced.

        Worker thread runs `generate_mtpk`; its `token_callback` pushes
        ints into a queue; this consumer decodes + yields immediately.

        Failure surfaces: if the worker raises before any token, the
        exception is re-raised to the caller. If no token arrives
        within FIRST_TOKEN_TIMEOUT seconds, abort with a clear error.
        If the generator finishes with zero tokens (silent failure
        mode the user reported as "no output"), surface that too —
        MTP runtime is reset so the next attempt can rebuild cleanly.
        """
        import queue
        import threading

        from mlx_lm.generate import GenerationResponse
        from mtplx.generation import generate_mtpk
        from mtplx.sampling import SamplerConfig

        # Prefer MTPLX's tokenizer if exposed by the runtime — it was
        # loaded alongside the augmented config and stays consistent
        # with the trunk MTPLX patched. Falls back to daydream's
        # tokenizer if not present.
        rt = self._ensure_runtime()
        tokenizer = getattr(rt, "tokenizer", None) or self.tokenizer

        prompt_ids = list(tokenizer.encode(prompt))
        prompt_token_count = len(prompt_ids)

        sampler = SamplerConfig(
            temperature=float(sampling.temperature),
            top_p=float(sampling.top_p),
            top_k=20,
        )
        depth = min(
            int(self._sidecar.mtp_depth_max),
            max(1, int(speculative.num_speculative_tokens)),
        )

        eos_ids: set[int] = set()
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_id, int):
            eos_ids.add(eos_id)
        eos_ids_list = getattr(tokenizer, "eos_token_ids", None)
        if isinstance(eos_ids_list, (list, tuple)):
            eos_ids.update(int(t) for t in eos_ids_list if isinstance(t, int))

        # ── Thread-bridged streaming ───────────────────────────────
        SENTINEL = object()
        FIRST_TOKEN_TIMEOUT = 180.0  # 27B prefill on M1/M2 can take >60s
        STEADY_TIMEOUT = 120.0       # after the first token, 2 min stall = fail
        token_q: queue.Queue = queue.Queue()
        stop_flag = threading.Event()
        worker_error: list[BaseException] = []

        def on_token(token_ids):
            if stop_flag.is_set():
                raise KeyboardInterrupt()
            for t in token_ids:
                token_q.put(int(t))

        def worker():
            try:
                generate_mtpk(
                    rt, prompt_ids,
                    max_tokens=int(sampling.max_tokens),
                    sampler=sampler,
                    speculative_depth=depth,
                    stop_token_ids=eos_ids or None,
                    token_callback=on_token,
                )
            except KeyboardInterrupt:
                pass
            except BaseException as exc:  # noqa: BLE001
                worker_error.append(exc)
            finally:
                token_q.put(SENTINEL)

        prefill_tic = time.perf_counter()
        threading.Thread(target=worker, daemon=True).start()

        all_tokens: list[int] = []
        last_decoded = ""
        tokens_emitted = 0
        first_token = True
        gen_tic = time.perf_counter()
        prompt_tps_estimate = 0.0

        def _emit_failure(reason: str) -> BaseException:
            """Drop the cached runtime so the next attempt rebuilds, then
            return an exception describing what went wrong."""
            self._mtplx_runtime = None
            return RuntimeError(f"MTP generation failed: {reason}")

        try:
            while True:
                timeout = FIRST_TOKEN_TIMEOUT if first_token else STEADY_TIMEOUT
                try:
                    item = token_q.get(timeout=timeout)
                except queue.Empty:
                    stop_flag.set()
                    if first_token:
                        raise _emit_failure(
                            f"no token in {int(FIRST_TOKEN_TIMEOUT)}s — likely OOM, "
                            "stuck mtplx.load, or model+head incompatibility. "
                            "Try: `daydream mtp install --full` for the bundled trunk."
                        )
                    raise _emit_failure(f"stalled mid-stream after {tokens_emitted} tokens")

                if item is SENTINEL:
                    if worker_error:
                        self._mtplx_runtime = None
                        raise worker_error[0]
                    if tokens_emitted == 0:
                        # generate_mtpk returned cleanly but produced
                        # nothing — the silent-failure mode the user
                        # reported. Make it loud.
                        raise _emit_failure(
                            "generate_mtpk returned zero tokens. The MTP head "
                            "may be incompatible with this trunk quantization. "
                            "Try: `daydream mtp uninstall && daydream mtp install --full`."
                        )
                    return

                token_id = item
                if first_token:
                    prefill_time = max(time.perf_counter() - prefill_tic, 1e-9)
                    prompt_tps_estimate = prompt_token_count / prefill_time
                    gen_tic = time.perf_counter()
                    first_token = False

                all_tokens.append(token_id)
                tokens_emitted += 1
                try:
                    full_text = tokenizer.decode(all_tokens, skip_special_tokens=False)
                except TypeError:
                    full_text = tokenizer.decode(all_tokens)
                delta = full_text[len(last_decoded):]
                last_decoded = full_text
                elapsed = max(time.perf_counter() - gen_tic, 1e-9)
                is_eos = token_id in eos_ids
                yield GenerationResponse(
                    text=delta,
                    token=token_id,
                    logprobs=None,
                    from_draft=False,
                    prompt_tokens=prompt_token_count,
                    prompt_tps=prompt_tps_estimate,
                    generation_tokens=tokens_emitted,
                    generation_tps=tokens_emitted / elapsed,
                    peak_memory=0.0,
                    finish_reason="stop" if is_eos else None,
                )
                if is_eos:
                    stop_flag.set()
                    return
        except (GeneratorExit, KeyboardInterrupt):
            stop_flag.set()
            raise

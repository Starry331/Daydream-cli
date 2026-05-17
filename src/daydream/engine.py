"""Inference engine — thin wrapper over mlx-lm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional

from daydream.models import ensure_runtime_model, get_model_path, is_fixture_model


def _patch_arrays_cache_for_speculative():
    """No-op kept for import compatibility.

    Earlier versions monkey-patched `ArraysCache.is_trimmable -> True`
    and a checkpoint-based trim so mlx-lm would attempt speculative
    decoding on Qwen3.5 / Qwen3.6 hybrid models. That was unsound:

      * The cache only checkpoints state after `advance(1)`. Speculation's
        main-model batch call is `advance(K+1)` in one shot, leaving NO
        checkpoint at any intermediate position.
      * `_patched_trim` therefore silently no-ops when asked to trim
        back a partially-rejected batch — but the cache state HAS
        advanced. The recurrent GatedDeltaNet state is now wrong for
        what mlx-lm thinks the position is, and every subsequent token
        is sampled from corrupted hidden state.
      * User-visible symptom: missing words, truncated phrases,
        broken sentences in the main body / reasoning chain (not a
        token-flood the degen detector can catch).

    The correct posture is to leave `ArraysCache.is_trimmable=False`
    and refuse external-draft speculation upstream — `_resolve_
    speculative_settings` / the /draft slash command both check
    `is_qwen_hybrid_runtime_model` and surface a clear message.
    """
    return

# Module-level cache for loaded models
_loaded_entries: dict[str, tuple[object, object]] = {}

# Module-level cache for MTP backends. The MTPLXRuntime each one
# wraps is ~16 GB of MLX weights — we MUST NOT rebuild it on every
# `engine.generate_stream` call. Keyed by `(model_ref, sidecar_path)`
# so the same backend instance survives across turns in a chat REPL.
_mtp_backends: dict[tuple[str, str], object] = {}


def get_or_create_mtp_backend(model, tokenizer, *, model_ref: str | None):
    """Return a cached MTPBackend for this (model_ref, sidecar) pair,
    or build one on first use.

    The cache key intentionally folds the sidecar path in too — if
    the user reinstalls MTP into a different dir between calls, we
    rebuild rather than reuse stale state.
    """
    from daydream.backends import MTPBackend
    from daydream.backends.mtp import detect_mtp_sidecar

    sidecar = detect_mtp_sidecar(model_ref)
    sidecar_path = str(sidecar.runtime_json_path) if sidecar is not None else ""
    cache_key = (str(model_ref or ""), sidecar_path)

    cached = _mtp_backends.get(cache_key)
    if cached is not None:
        return cached

    backend = MTPBackend(model, tokenizer, model_ref=model_ref)
    backend.prepare()
    _mtp_backends[cache_key] = backend
    return backend


def teardown_mtp_backends() -> None:
    """Release all cached MTPBackend instances (frees their MTPLXRuntime
    objects so MLX can reclaim GPU memory).

    Called by chat.py when the user switches *out* of MTP mode via
    /draft off or /draft on — without this, the 16 GB MTP trunk stays
    pinned in memory for the rest of the session.
    """
    import gc

    for backend in list(_mtp_backends.values()):
        try:
            backend.teardown()
        except Exception:
            pass
    _mtp_backends.clear()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass
    gc.collect()


@dataclass
class FixtureModel:
    repo_id: str


@dataclass
class FixtureTokenizer:
    repo_id: str

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        parts = [f"{message['role']}: {message.get('content', '')}" for message in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        prompt = "\n".join(parts)
        if tokenize:
            return prompt.split()
        return prompt


@dataclass
class FixtureResponse:
    text: str
    token: int
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: Optional[str]


def speculative_runtime_status(model) -> tuple[bool, str | None]:
    """Check whether the loaded model can use speculative decoding safely.

    We intentionally avoid blocking draft mode up front. Some mlx-lm model
    families have runtime/cache behaviors that are only knowable when the
    generation path actually runs. Daydream enables draft mode optimistically
    and lets the generation/server fallback path handle real incompatibilities.
    """
    return True, None


def speculative_server_status(
    model,
    tokenizer,
    draft_model,
    *,
    num_draft_tokens: int = 6,
) -> tuple[bool, str | None]:
    """Probe speculative decoding through the same prompt-cache path mlx-lm server uses."""
    if isinstance(model, FixtureModel):
        return True, None
    if draft_model is None:
        return False, "Draft acceleration requires a loaded draft model."
    module_name = type(model).__module__
    if not module_name.startswith("mlx_lm."):
        return True, None

    try:
        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "hi"}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "user: hi\nassistant:"

        prompt_cache = make_prompt_cache(model) + make_prompt_cache(draft_model)
        sampler = make_sampler(temp=0.0, top_p=1.0)
        generator = mlx_stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=1,
            sampler=sampler,
            prompt_cache=prompt_cache,
            draft_model=draft_model,
            num_draft_tokens=num_draft_tokens,
        )
        next(generator, None)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _fixture_reply(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", "")).lower()
        if "hello" in content:
            return "Hello from Daydream."
        return "Daydream is running in offline fixture mode."
    return "Hello from Daydream."


def load_model(name: str, verbose: bool = False, *, ensure_available: bool = True):
    """Load a model by short name or HF repo ID. Caches the result."""
    repo_id = (
        ensure_runtime_model(name, auto_pull=True, register_alias=True)
        if ensure_available
        else name
    )
    if repo_id in _loaded_entries:
        return _loaded_entries[repo_id]

    if is_fixture_model(repo_id):
        entry = (FixtureModel(repo_id), FixtureTokenizer(repo_id))
        _loaded_entries[repo_id] = entry
        return entry

    if verbose:
        from rich.console import Console

        Console(stderr=True).print(f"[dim]Loading {repo_id}...[/dim]")

    from mlx_lm import load as mlx_load

    model_ref = str(get_model_path(repo_id) or repo_id)
    entry = mlx_load(model_ref)
    _loaded_entries[repo_id] = entry
    return entry


def warmup_draft_model(model, draft_model, tokenizer) -> None:
    """Compile Metal shaders for both models WITHOUT running them jointly.

    Running mlx_stream_generate(main, draft=draft) on a hybrid pair
    (Qwen3.5 / Qwen3.6) constructs a fused command buffer that frequently
    exceeds Metal's per-buffer execution timeout
    (kIOGPUCommandBufferCallbackErrorTimeout) on machines with tight
    unified-memory headroom — and that surfaces as a C++ std::abort
    which Python cannot recover from.

    We avoid that landmine by warming each model on its own. The first
    real chat token after this will JIT-compile the joint kernels, but
    it'll be inside the existing generate_stream try/except (which can
    fall back to plain decoding without crashing the process).
    """
    if isinstance(model, FixtureModel):
        return

    try:
        import mlx.core as mx
        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        return

    sampler = make_sampler(temp=0.0)

    def _warmup_single(target) -> None:
        try:
            for _ in mlx_stream_generate(
                target, tokenizer, prompt="hi", max_tokens=1, sampler=sampler,
            ):
                pass
        except Exception:
            pass

    _warmup_single(model)
    if draft_model is not None and draft_model is not model:
        _warmup_single(draft_model)
    try:
        mx.clear_cache()
    except Exception:
        pass


def create_prompt_cache(model, draft_model=None, *, max_kv_size: int | None = None):
    """Create a reusable prompt cache for chat turns.

    `max_kv_size` caps the working KV window in tokens. Once the cache
    has that many entries, mlx-lm rotates them in-place. None means
    "model default" (the model's max_position_embeddings).
    """
    if isinstance(model, FixtureModel):
        return None
    if not type(model).__module__.startswith("mlx_lm."):
        return None
    try:
        from mlx_lm.models.cache import make_prompt_cache

        prompt_cache = make_prompt_cache(model, max_kv_size=max_kv_size) if max_kv_size else make_prompt_cache(model)
        if draft_model is not None:
            draft_cache = make_prompt_cache(draft_model, max_kv_size=max_kv_size) if max_kv_size else make_prompt_cache(draft_model)
            prompt_cache += draft_cache
        return prompt_cache
    except Exception:
        return None


def model_max_position_embeddings(model) -> int | None:
    """Return the loaded model's positional-embedding limit, if known."""
    if isinstance(model, FixtureModel):
        return None
    args = getattr(model, "args", None)
    for attr in ("max_position_embeddings", "max_position_embedding"):
        value = getattr(args, attr, None) if args is not None else None
        if isinstance(value, int) and value > 0:
            return value
    cfg = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "max_position_embedding"):
        value = getattr(cfg, attr, None) if cfg is not None else None
        if isinstance(value, int) and value > 0:
            return value
    return None


def available_ram_bytes() -> Optional[int]:
    """Best-effort estimate of immediately-allocatable RAM in bytes.

    Apple Silicon uses unified memory, so 'free RAM' is the cap on what
    MLX can put on the GPU. We parse `vm_stat` because macOS doesn't
    ship psutil — and we want this check to run without adding a
    runtime dependency.

    Returns None if probing fails — callers must treat that as
    'unknown, don't refuse'.
    """
    import subprocess

    try:
        page_size = 16384  # M1/M2/M3 default; verified below.
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=2,
        )
        if out.returncode != 0:
            return None
        free_pages = inactive_pages = speculative_pages = 0
        for line in out.stdout.splitlines():
            line = line.strip()
            if line.startswith("Mach Virtual Memory Statistics") and "page size of" in line:
                try:
                    page_size = int(line.split("page size of")[1].split()[0])
                except Exception:
                    pass
            elif line.startswith("Pages free"):
                free_pages = int(line.split(":")[1].strip().rstrip("."))
            elif line.startswith("Pages inactive"):
                inactive_pages = int(line.split(":")[1].strip().rstrip("."))
            elif line.startswith("Pages speculative"):
                speculative_pages = int(line.split(":")[1].strip().rstrip("."))
        return (free_pages + inactive_pages + speculative_pages) * page_size
    except Exception:
        return None


def set_metal_wired_limit() -> None:
    """Set Metal GPU wired memory limit to maximum for optimal performance."""
    try:
        import mlx.core as mx

        if mx.metal.is_available():
            max_size = mx.device_info()["max_recommended_working_set_size"]
            mx.set_wired_limit(max_size)
    except Exception:
        pass


def _model_needs_chunked_speculative(model) -> bool:
    """Check if the model's cache includes ArraysCache layers that only have
    approximate trim (via our monkey-patch).  These models need the draft-aware
    generation path."""
    if isinstance(model, FixtureModel):
        return False
    try:
        from mlx_lm.models.cache import ArraysCache, make_prompt_cache
        cache = make_prompt_cache(model)
        result = any(isinstance(c, ArraysCache) for c in cache)
        del cache
        return result
    except Exception:
        return False


def _draft_with_fallback(
    model, tokenizer, draft_model, prompt_str, *,
    max_tokens, sampler, num_draft_tokens=2, prefill_step_size=None,
    prompt_cache=None,
):
    """Draft-accelerated generation with a short safety buffer.

    On hybrid models (Qwen3.5 / Qwen3.6) the GatedDeltaNet recurrent
    state cannot be cleanly rewound after a rejected draft token. To
    avoid showing the user any garbage we use a *short* probe window
    at the start of generation — if the draft pair degenerates in those
    first few tokens we discard them and replay through the main model
    alone. After the probe window passes we stream tokens immediately,
    so TTFT stays close to the underlying draft path.

    Detection: single-token flood (5/10) OR bigram flood (4/12).
    """
    import time
    from collections import Counter

    import mlx.core as mx
    from mlx_lm import stream_generate as mlx_stream_generate
    from mlx_lm.generate import GenerationResponse

    prompt_tokens = mx.array(tokenizer.encode(prompt_str))
    tic = time.perf_counter()

    # Degeneration detector state. Tightened thresholds vs the historic
    # 8/15 + 6/20 — those required 20 buffered tokens before deciding,
    # which made TTFT roughly the same as plain generation. The smaller
    # windows still catch the dominant failure mode (the model getting
    # stuck on a token / token-pair) within ~10 tokens.
    _recent: list[int] = []
    _bigrams: list[tuple[int, int]] = []

    def _is_degenerate() -> bool:
        # Single-token flood: 5 copies of the same token in last 10 emitted.
        if len(_recent) >= 10:
            if Counter(_recent[-10:]).most_common(1)[0][1] >= 5:
                return True
        # Bigram flood (catches "* * * *" alternation). The 4-in-12 rule
        # only fires after 13 tokens, which doesn't help if the probe
        # phase (8 tokens) already shows alternation. Add a tighter
        # early-probe rule: 3 copies of the same bigram in the last 6.
        if len(_bigrams) >= 6:
            if Counter(_bigrams[-6:]).most_common(1)[0][1] >= 3:
                return True
        if len(_bigrams) >= 12:
            if Counter(_bigrams[-12:]).most_common(1)[0][1] >= 4:
                return True
        return False

    # ── Phase 1: short probe (≤ PROBE_TOKENS, buffered for rollback) ──
    # Buffer a handful of tokens so we can throw them away if the
    # draft+main combo degenerates immediately. Anything beyond this
    # streams live — at 50 tok/s, PROBE_TOKENS=8 means the user sees
    # output after ~150 ms, which is comparable to non-draft TTFT.
    PROBE_TOKENS = 8
    buffered: list[GenerationResponse] = []
    prompt_tps: float = 0.0
    degenerated = False

    extra_kwargs: dict = {
        "draft_model": draft_model,
        "num_draft_tokens": num_draft_tokens,
    }
    if prefill_step_size is not None:
        extra_kwargs["prefill_step_size"] = prefill_step_size
    if prompt_cache is not None:
        extra_kwargs["prompt_cache"] = prompt_cache

    stream = mlx_stream_generate(
        model, tokenizer, prompt_str,
        max_tokens=max_tokens, sampler=sampler,
        **extra_kwargs,
    )

    # Probe phase: keep tokens in a buffer so we can fall back cleanly.
    for response in stream:
        _recent.append(response.token)
        if len(_recent) > 1:
            _bigrams.append((_recent[-2], _recent[-1]))
        if not buffered:
            prompt_tps = response.prompt_tps
        buffered.append(response)

        if response.finish_reason == "stop":
            # Short answer that finished inside the probe — yield buffer.
            break

        if _is_degenerate():
            degenerated = True
            break

        if len(buffered) >= PROBE_TOKENS:
            break  # transition to live streaming

    if not degenerated:
        # Yield the buffered probe tokens, then continue streaming live.
        for i, response in enumerate(buffered):
            yield GenerationResponse(
                text=response.text,
                token=response.token,
                logprobs=response.logprobs,
                from_draft=response.from_draft,
                prompt_tokens=prompt_tokens.size,
                prompt_tps=prompt_tps,
                generation_tokens=i + 1,
                generation_tps=(i + 1) / (time.perf_counter() - tic) if (i + 1) > 0 else 0.0,
                peak_memory=response.peak_memory,
                finish_reason=response.finish_reason,
            )
            if response.finish_reason == "stop":
                return

        # ── Phase 1b: live stream from the same draft path ─────────────
        # We're past the probe and the model isn't degenerate. Continue
        # consuming the SAME stream generator (so the spec-decoding KV
        # state is preserved) and emit tokens as they arrive.
        #
        # Critical: we KEEP running the degeneration detector every
        # token, not just during the probe. If the draft pair flips
        # into a repetition loop mid-generation (the classic
        # "* * * * *" or "8 8 8 8" failure mode), we stop emitting
        # rather than let the user watch an asterisk field scroll by.
        ntoks = len(buffered)
        for response in stream:
            _recent.append(response.token)
            if len(_recent) > 1:
                _bigrams.append((_recent[-2], _recent[-1]))
            # Trim sliding windows so memory doesn't grow unbounded.
            if len(_recent) > 32:
                _recent = _recent[-32:]
            if len(_bigrams) > 32:
                _bigrams = _bigrams[-32:]
            ntoks += 1
            yield GenerationResponse(
                text=response.text,
                token=response.token,
                logprobs=response.logprobs,
                from_draft=response.from_draft,
                prompt_tokens=prompt_tokens.size,
                prompt_tps=prompt_tps,
                generation_tokens=ntoks,
                generation_tps=ntoks / (time.perf_counter() - tic) if ntoks > 0 else 0.0,
                peak_memory=response.peak_memory,
                finish_reason=response.finish_reason,
            )
            if response.finish_reason:
                return
            if _is_degenerate():
                # Stop emitting — the loop is real, not transient.
                # We can't rewind the tokens already yielded, but at
                # least the chat reply doesn't fill the screen.
                try:
                    stream.close()
                except Exception:
                    pass
                return
        return

    # ── Phase 2: draft degenerated → discard buffer, replay clean ────
    # Drain the generator so mlx-lm finalizes its internal state, then
    # restart on the main model alone with the same prompt.
    try:
        stream.close()
    except Exception:
        pass

    fallback_kwargs: dict = {}
    if prefill_step_size is not None:
        fallback_kwargs["prefill_step_size"] = prefill_step_size

    ntoks = 0
    for response in mlx_stream_generate(
        model, tokenizer, prompt_str,
        max_tokens=max_tokens, sampler=sampler,
        **fallback_kwargs,
    ):
        ntoks += 1
        yield GenerationResponse(
            text=response.text,
            token=response.token,
            logprobs=response.logprobs,
            from_draft=False,
            prompt_tokens=prompt_tokens.size,
            prompt_tps=prompt_tps or response.prompt_tps,
            generation_tokens=ntoks,
            generation_tps=ntoks / (time.perf_counter() - tic) if ntoks > 0 else 0.0,
            peak_memory=response.peak_memory,
            finish_reason=response.finish_reason,
        )
        if response.finish_reason:
            return


def generate_stream(
    model,
    tokenizer,
    messages: list[dict],
    *,
    max_tokens: int = 4096,
    temp: float = 0.6,
    top_p: float = 0.9,
    chat_template_kwargs: dict | None = None,
    draft_model=None,
    num_draft_tokens: int | None = None,
    prefill_step_size: int | None = None,
    prompt_cache=None,
    pld_enabled: bool = False,
    mtp_enabled: bool = False,
    mtp_model_ref: str | None = None,
) -> Generator:
    """Stream-generate a response from a list of chat messages."""
    if isinstance(model, FixtureModel):
        text = _fixture_reply(messages)
        yield FixtureResponse(
            text=text[:max_tokens],
            token=0,
            prompt_tokens=sum(len(str(m.get("content", "")).split()) for m in messages),
            prompt_tps=1.0,
            generation_tokens=len(text.split()),
            generation_tps=1.0,
            peak_memory=0.0,
            finish_reason="stop",
        )
        return

    # Apply chat template to get the prompt string.
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )
    else:
        prompt = "\n".join(f"{message['role']}: {message.get('content', '')}" for message in messages)

    from mlx_lm import stream_generate as mlx_stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temp, top_p=top_p)

    # MTP path: model-native multi-token prediction via the MTPBackend.
    # Uses a module-level cache so the 16 GB MTPLXRuntime survives
    # across chat turns instead of being rebuilt each time.
    if mtp_enabled and draft_model is None:
        from daydream.backends.base import SamplingParams, SpeculativeParams

        backend = get_or_create_mtp_backend(model, tokenizer, model_ref=mtp_model_ref)
        caps = backend.capabilities()
        if caps.supports_speculative:
            sampling = SamplingParams(temperature=temp, top_p=top_p, max_tokens=max_tokens)
            speculative = SpeculativeParams(
                method="mtp",
                num_speculative_tokens=num_draft_tokens or 2,
            )
            yield from backend.generate(
                messages,
                sampling,
                speculative,
                chat_template_kwargs=chat_template_kwargs,
                prompt_cache=prompt_cache,
                prefill_step_size=prefill_step_size,
            )
            return
        # else: fall through; the caller already saw the warning at
        # /draft / --speculative time.

    # PLD (prompt-lookup decoding) path: no draft model, ngram-based
    # speculation. Only safe on models with trimmable prompt caches.
    if pld_enabled and draft_model is None:
        from daydream.pld import pld_stream, supports_pld

        ok, _reason = supports_pld(model)
        if ok:
            yield from pld_stream(
                model, tokenizer, prompt,
                max_tokens=max_tokens, sampler=sampler,
                prompt_cache=prompt_cache,
            )
            return
        # If PLD isn't safe on this model, silently fall through to
        # plain decoding rather than corrupt output.

    # For models with non-trimmable caches (Qwen3.5 / Qwen3.6): run a
    # short safety probe under the draft, then stream live.
    if draft_model is not None and _model_needs_chunked_speculative(model):
        yield from _draft_with_fallback(
            model, tokenizer, draft_model, prompt,
            max_tokens=max_tokens, sampler=sampler,
            num_draft_tokens=num_draft_tokens or 2,
            prefill_step_size=prefill_step_size,
            prompt_cache=prompt_cache,
        )
        return

    try:
        extra_kwargs = {}
        if draft_model is not None:
            extra_kwargs["draft_model"] = draft_model
            if num_draft_tokens is not None:
                extra_kwargs["num_draft_tokens"] = num_draft_tokens
        if prefill_step_size is not None:
            extra_kwargs["prefill_step_size"] = prefill_step_size
        if prompt_cache is not None:
            extra_kwargs["prompt_cache"] = prompt_cache
        for response in mlx_stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            **extra_kwargs,
        ):
            yield response
    except Exception as exc:
        if draft_model is None:
            raise
        import sys

        print(
            f"\033[2m[draft fallback] Speculative decoding error: {exc}; continuing without draft.\033[0m",
            file=sys.stderr,
        )
        fallback_kwargs = {}
        if prefill_step_size is not None:
            fallback_kwargs["prefill_step_size"] = prefill_step_size
        for response in mlx_stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            **fallback_kwargs,
        ):
            yield response

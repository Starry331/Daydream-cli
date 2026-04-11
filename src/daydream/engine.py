"""Inference engine — thin wrapper over mlx-lm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional

from daydream.models import ensure_runtime_model, get_model_path, is_fixture_model


def _patch_arrays_cache_for_speculative():
    """Monkey-patch ArraysCache for speculative decoding on Qwen3.5.

    Draft model (advance(1) per token): exact checkpoint-based restore.
    Main model (advance(N) batch): no-op trim — the GatedDeltaNet recurrent
    state cannot be partially rewound, but the attention layers (KVCache)
    always have correct context.
    """
    try:
        from mlx_lm.models.cache import ArraysCache

        if ArraysCache(1).is_trimmable():
            return

        _original_init = ArraysCache.__init__
        _original_advance = ArraysCache.advance

        def _patched_init(self, size, left_padding=None):
            _original_init(self, size, left_padding=left_padding)
            self._checkpoints: dict[int, tuple] = {}
            self._position: int = 0

        def _patched_advance(self, N):
            _original_advance(self, N)
            self._position += N
            if N == 1:
                self._checkpoints[self._position] = (
                    list(self.cache), self.lengths, self.left_padding,
                )
                if len(self._checkpoints) > 12:
                    cutoff = self._position - 10
                    self._checkpoints = {
                        k: v for k, v in self._checkpoints.items() if k >= cutoff
                    }

        def _patched_is_trimmable(self):
            return True

        def _patched_trim(self, n):
            if n <= 0:
                return 0
            target = self._position - n
            if target in self._checkpoints:
                state, lengths, left_padding = self._checkpoints[target]
                self.cache = list(state)
                self.lengths = lengths
                self.left_padding = left_padding
                self._position = target
                self._checkpoints = {
                    k: v for k, v in self._checkpoints.items() if k <= target
                }
            # Main model batch path: no checkpoint → no-op (safe).
            return n

        ArraysCache.__init__ = _patched_init
        ArraysCache.advance = _patched_advance
        ArraysCache.is_trimmable = _patched_is_trimmable
        ArraysCache.trim = _patched_trim
    except ImportError:
        pass


_patch_arrays_cache_for_speculative()

# Module-level cache for loaded models
_loaded_entries: dict[str, tuple[object, object]] = {}


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
    """Run a tiny speculative generation to compile Metal shaders for both models."""
    if isinstance(model, FixtureModel):
        return
    try:
        import mlx.core as mx
        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=0.0)
        # Warm up both models together via the speculative path
        for _ in mlx_stream_generate(
            model, tokenizer, prompt="hi", max_tokens=2, sampler=sampler,
            draft_model=draft_model, num_draft_tokens=2,
        ):
            pass
        mx.clear_cache()
    except Exception:
        # Fallback: warm up individually
        try:
            for _ in mlx_stream_generate(
                model, tokenizer, prompt="hi", max_tokens=1, sampler=sampler,
            ):
                pass
            mx.clear_cache()
        except Exception:
            pass


def create_prompt_cache(model, draft_model=None):
    """Create a reusable prompt cache for chat turns."""
    if isinstance(model, FixtureModel):
        return None
    if not type(model).__module__.startswith("mlx_lm."):
        return None
    try:
        from mlx_lm.models.cache import make_prompt_cache

        prompt_cache = make_prompt_cache(model)
        if draft_model is not None:
            prompt_cache += make_prompt_cache(draft_model)
        return prompt_cache
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
    max_tokens, sampler, num_draft_tokens=2,
):
    """Draft-accelerated generation with degeneration fallback.

    Runs mlx-lm's native speculative loop at full speed.  Monitors every
    token for degeneration patterns.  If detected, discards ALL draft output
    and restarts cleanly with the main model from the original prompt.

    Detection: single-token flood (8/15) OR bigram flood (6/20).
    """
    import time
    from collections import Counter

    import mlx.core as mx
    from mlx_lm import stream_generate as mlx_stream_generate
    from mlx_lm.generate import GenerationResponse

    prompt_tokens = mx.array(tokenizer.encode(prompt_str))
    tic = time.perf_counter()

    # Degeneration detector state
    _recent: list[int] = []
    _bigrams: list[tuple[int, int]] = []

    def _is_degenerate() -> bool:
        # Single token flood: 8+ of same token in last 15
        if len(_recent) >= 15:
            if Counter(_recent[-15:]).most_common(1)[0][1] >= 8:
                return True
        # Bigram flood: same 2-token pair 6+ times in last 20 bigrams
        if len(_bigrams) >= 20:
            if Counter(_bigrams[-20:]).most_common(1)[0][1] >= 6:
                return True
        return False

    # ── Phase 1: Draft generation (buffered, short — max 50 tokens) ──
    # Only buffer a small number of tokens to limit the "blank" period.
    # If the model finishes (EOS) within 50 tokens → yield buffer (fast).
    # If not or degenerated → discard, stream with normal model (Phase 2).
    buffered: list[GenerationResponse] = []
    prompt_tps: float = 0.0
    degenerated = False
    draft_cap = min(30, max_tokens)

    for response in mlx_stream_generate(
        model, tokenizer, prompt_str,
        max_tokens=draft_cap, sampler=sampler,
        draft_model=draft_model, num_draft_tokens=num_draft_tokens,
    ):
        _recent.append(response.token)
        if len(_recent) > 1:
            _bigrams.append((_recent[-2], _recent[-1]))
        if len(buffered) == 0:
            prompt_tps = response.prompt_tps
        buffered.append(response)

        if response.finish_reason == "stop":
            break

        if _is_degenerate():
            degenerated = True
            break

    # If draft completed with EOS and no degeneration → yield buffer (fast path)
    phase1_eos = buffered and buffered[-1].finish_reason == "stop"
    if not degenerated and phase1_eos:
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
        return
    # Otherwise (degenerated OR didn't finish in 50 tokens) → Phase 2

    # ── Phase 2: Draft degenerated → discard buffer, restart with main model ──
    ntoks = 0
    for response in mlx_stream_generate(
        model, tokenizer, prompt_str,
        max_tokens=max_tokens, sampler=sampler,
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

    # For models with non-trimmable caches (Qwen3.5): run draft first,
    # detect degeneration, fallback to clean normal generation if needed.
    if draft_model is not None and _model_needs_chunked_speculative(model):
        yield from _draft_with_fallback(
            model, tokenizer, draft_model, prompt,
            max_tokens=max_tokens, sampler=sampler,
            num_draft_tokens=num_draft_tokens or 2,
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

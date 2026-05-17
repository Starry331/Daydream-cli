"""Backend abstraction for Daydream generation paths.

A backend bundles three responsibilities:

1. **Loading**:  Decide what to load (main + optional sidecar/draft).
2. **Capability**: Report what it supports so the CLI can validate
   user choices up front, before the model is loaded.
3. **Generation**: Stream tokens given (model, tokenizer, messages,
   sampling, speculative-params).

The dispatcher (`backend_for`) picks a backend at run-time based on
the resolved speculative method. Phase 1 already shipped the resolver
(`daydream.speculative_methods.resolve_method`); Phase 2 plugs concrete
backends into it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Generator, Optional


@dataclass(frozen=True)
class SamplingParams:
    """Per-request sampling knobs."""

    temperature: float = 0.6
    top_p: float = 0.9
    max_tokens: int = 4096


@dataclass(frozen=True)
class SpeculativeParams:
    """Per-request speculative-decoding knobs."""

    method: str = "none"  # auto / mtp / draft / lookup / none
    num_speculative_tokens: int = 2
    draft_model_repo: Optional[str] = None
    # Future: kv_bits, prefill_step_size, etc. — kept here so backends
    # share a single param object instead of growing per-backend
    # keyword arguments.


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend can do for the model it was bound to.

    `available` here means "the backend can actually run this method
    on the model that was passed to `prepare()`". For example,
    `MTPBackend(qwen36_no_sidecar)` will report `supports_speculative
    = False` because the sidecar weights weren't found.
    """

    method: str
    supports_speculative: bool
    external_weights_required: bool = False
    notes: str = ""
    diagnostic: dict[str, Any] = field(default_factory=dict)


class GenerationBackend(abc.ABC):
    """Abstract base class for all Daydream generation backends.

    Lifecycle:
        backend = SomeBackend(model, tokenizer)
        backend.prepare()                  # load sidecar weights, JIT-compile shaders, etc.
        caps = backend.capabilities()      # query support BEFORE running
        for response in backend.generate(messages, sampling, speculative):
            ...
    """

    method: str = "unspecified"

    def __init__(self, model, tokenizer, *, model_ref: Optional[str] = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_ref = model_ref
        self._prepared = False

    # ── Lifecycle ────────────────────────────────────────────────
    def prepare(self) -> None:
        """One-time setup. Subclasses override for sidecar loads, warmup, etc."""
        self._prepared = True

    def teardown(self) -> None:  # noqa: B027 - intentional empty hook
        """Release sidecar weights / caches. Default: noop."""

    # ── Capability reporting ────────────────────────────────────
    @abc.abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return what this backend offers for its bound model."""

    # ── Generation ──────────────────────────────────────────────
    @abc.abstractmethod
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
        """Stream `GenerationResponse`-shaped objects (see daydream.engine)."""


def backend_for(method: str, model, tokenizer, *, model_ref: Optional[str] = None) -> GenerationBackend:
    """Construct the right backend for the requested speculative method.

    `method` is the *resolved* method from
    `daydream.speculative_methods.resolve_method` — NOT the raw user
    input. Resolution should happen at the CLI layer; backends are
    the runtime layer that executes the decision.
    """
    from . import lookup, mlx_native, mtp

    if method == "mtp":
        return mtp.MTPBackend(model, tokenizer, model_ref=model_ref)
    if method == "lookup":
        return lookup.LookupBackend(model, tokenizer, model_ref=model_ref)
    return mlx_native.MLXBackend(model, tokenizer, model_ref=model_ref)

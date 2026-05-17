"""Prompt-lookup decoding backend — wraps `daydream.pld.pld_stream`.

No second model, no extra memory. Effective on prompts where the
output overlaps with the input (code refactors, doc edits, long
quotes); free on chat. Falls back transparently to single-token
decoding when no ngram match is found.

Hard constraint: requires a trimmable prompt cache. `Qwen3.5` /
`Qwen3.6` use a hybrid GatedDeltaNet cache that cannot be cleanly
trimmed — the backend's `capabilities()` reports
`supports_speculative=False` in that case so the dispatcher can
degrade gracefully.
"""

from __future__ import annotations

from typing import Any, Generator, Optional

from .base import BackendCapabilities, GenerationBackend, SamplingParams, SpeculativeParams


class LookupBackend(GenerationBackend):
    method = "lookup"

    def capabilities(self) -> BackendCapabilities:
        from daydream.pld import supports_pld

        ok, reason = supports_pld(self.model)
        return BackendCapabilities(
            method="lookup",
            supports_speculative=ok,
            external_weights_required=False,
            notes=(
                "Prompt-lookup decoding (PLD). No second model. Free "
                "on chat; ~2-3× on code/refactor prompts."
                if ok else
                f"PLD unavailable on this model: {reason}"
            ),
        )

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
        from daydream import engine

        # Reuse engine.generate_stream's `pld_enabled` plumbing so the
        # caller gets the same retry / fallback envelope as direct
        # callers of engine.generate_stream.
        yield from engine.generate_stream(
            self.model,
            self.tokenizer,
            messages,
            max_tokens=sampling.max_tokens,
            temp=sampling.temperature,
            top_p=sampling.top_p,
            chat_template_kwargs=chat_template_kwargs,
            draft_model=None,
            num_draft_tokens=None,
            prefill_step_size=prefill_step_size,
            prompt_cache=prompt_cache,
            pld_enabled=True,
        )

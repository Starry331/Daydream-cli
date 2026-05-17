"""MLX-native backend — wraps `daydream.engine.generate_stream`.

Covers both the no-speculation path AND the external-draft-model path.
The split between "no draft" and "external draft" is just a parameter
passed to `engine.generate_stream`; this backend is the same code
path either way.

Used for:
    --speculative none
    --speculative draft   (with `speculative.draft_model_repo` set)
    --speculative auto    (when the resolver picks `draft` or `none`)
"""

from __future__ import annotations

from typing import Any, Generator, Optional

from .base import BackendCapabilities, GenerationBackend, SamplingParams, SpeculativeParams


class MLXBackend(GenerationBackend):
    method = "mlx-native"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            method="mlx-native",
            supports_speculative=True,  # via external draft model
            external_weights_required=False,
            notes=(
                "Standard mlx-lm path. With speculative.draft_model_repo "
                "set, uses external-draft speculative decoding. Without "
                "it, regular greedy/sampled decoding."
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
        # Lazy import — engine.py pulls in mlx_lm at module load time,
        # which we don't want to do unless we're actually going to run.
        from daydream import engine

        draft_model = None
        if speculative.method == "draft" and speculative.draft_model_repo:
            draft_model = engine.load_model(speculative.draft_model_repo, ensure_available=False)[0]

        yield from engine.generate_stream(
            self.model,
            self.tokenizer,
            messages,
            max_tokens=sampling.max_tokens,
            temp=sampling.temperature,
            top_p=sampling.top_p,
            chat_template_kwargs=chat_template_kwargs,
            draft_model=draft_model,
            num_draft_tokens=(
                speculative.num_speculative_tokens
                if speculative.method == "draft"
                else None
            ),
            prefill_step_size=prefill_step_size,
            prompt_cache=prompt_cache,
        )

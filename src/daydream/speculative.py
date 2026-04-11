"""Speculative decoding helpers."""

from __future__ import annotations

from pathlib import Path

QWEN35_DRAFT_MODEL = "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
QWEN35_NUM_DRAFT_TOKENS = 2


def _normalized_hint(value: str | None) -> str:
    if not value:
        return ""
    return Path(str(value)).name.lower()


def is_qwen35_runtime_model(value: str | None) -> bool:
    """Return True when the model reference looks like a Qwen3.5 MLX model."""
    lowered = str(value or "").lower()
    hint = _normalized_hint(value)
    return "qwen3.5" in lowered or "qwen3.5" in hint


def default_draft_for_model(value: str | None) -> str | None:
    """Return the built-in draft model for auto-enabling draft acceleration.

    Qwen3.5's GatedDeltaNet recurrent state (ArraysCache) cannot be
    properly rewound after rejected draft tokens, causing the main model
    to degenerate into repetitive/garbled reasoning.  Draft acceleration
    is disabled until mlx-lm adds native ArraysCache trim support.
    """
    return None


def draft_model_for_family(value: str | None) -> str | None:
    """Return the draft model repo for a given model family."""
    if is_qwen35_runtime_model(value):
        return QWEN35_DRAFT_MODEL
    return None


def default_num_draft_tokens(value: str | None) -> int | None:
    if not is_qwen35_runtime_model(value):
        return None
    return QWEN35_NUM_DRAFT_TOKENS


def supports_manual_draft(value: str | None) -> bool:
    return draft_model_for_family(value) is not None

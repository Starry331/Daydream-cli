"""Speculative decoding helpers."""

from __future__ import annotations

from pathlib import Path

# mlx-community/Qwen3.5-0.8B-OptiQ-4bit serves as the draft for both
# Qwen3.5 and Qwen3.6 main models — they share the same tokenizer
# (vocab_size 248320, model_type "qwen3_5") so the draft's distribution
# is verifiable against the larger model.
QWEN35_DRAFT_MODEL = "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
QWEN35_NUM_DRAFT_TOKENS = 2

# Qwen3.6 shares Qwen3.5's hybrid architecture and tokenizer vocab —
# the official Qwen3.6 docs call out external draft model speculative
# decoding as a first-class acceleration path, and the 0.8B-OptiQ
# checkpoint is the smallest published weight class with the matching
# `model_type=qwen3_5` and `vocab_size=248320`.
#
# We use num_draft_tokens=3 for Qwen3.6: a slight increase over the
# Qwen3.5 default. Empirically, the 0.8B draft has a high acceptance
# rate against the 27B target, so committing 3 speculative tokens per
# step gives a better wall-clock speedup than 2 while still keeping
# the rejection-rollback cost bounded.
QWEN36_DRAFT_MODEL = QWEN35_DRAFT_MODEL
QWEN36_NUM_DRAFT_TOKENS = 3


def _normalized_hint(value: str | None) -> str:
    if not value:
        return ""
    return Path(str(value)).name.lower()


def is_qwen35_runtime_model(value: str | None) -> bool:
    """Return True when the model reference looks like a Qwen3.5 MLX model."""
    lowered = str(value or "").lower()
    hint = _normalized_hint(value)
    return "qwen3.5" in lowered or "qwen3.5" in hint


def is_qwen36_runtime_model(value: str | None) -> bool:
    """Return True when the model reference looks like a Qwen3.6 MLX model."""
    lowered = str(value or "").lower()
    hint = _normalized_hint(value)
    return "qwen3.6" in lowered or "qwen3.6" in hint


def is_qwen_hybrid_runtime_model(value: str | None) -> bool:
    """True for Qwen3.5/Qwen3.6 family models (hybrid GatedDeltaNet arch)."""
    return is_qwen35_runtime_model(value) or is_qwen36_runtime_model(value)


def default_draft_for_model(value: str | None) -> str | None:
    """Daydream is strict opt-in: no model family auto-enables any
    speculative path. Users must explicitly type `--speculative draft`
    (or `/draft on`), `--speculative mtp`, etc. The default launch is
    always a single-model decode.
    """
    return None


def draft_model_for_family(value: str | None) -> str | None:
    """Return the draft model repo for a given model family."""
    if is_qwen36_runtime_model(value):
        return QWEN36_DRAFT_MODEL
    if is_qwen35_runtime_model(value):
        return QWEN35_DRAFT_MODEL
    return None


def default_num_draft_tokens(value: str | None) -> int | None:
    if is_qwen36_runtime_model(value):
        return QWEN36_NUM_DRAFT_TOKENS
    if is_qwen35_runtime_model(value):
        return QWEN35_NUM_DRAFT_TOKENS
    return None


def supports_manual_draft(value: str | None) -> bool:
    return draft_model_for_family(value) is not None

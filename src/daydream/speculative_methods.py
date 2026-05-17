"""Speculative-decoding method dispatch + per-model capability table.

Daydream supports several speculative-decoding strategies. They share
one user-facing knob (`--speculative` / `/draft`) but route to different
backends internally.

Methods
-------
- ``auto``    — pick the best *currently working* method for this model.
- ``mtp``     — Multi-Token Prediction using the target model's built-in
                MTP heads (Qwen3.6's official path). **Not yet wired**:
                mlx-lm 0.31.2 strips MTP weights at load time, so this
                method currently raises ``MTPNotAvailable``. Tracked as
                Phase 2 work — will route to MTPLX or a forked mlx-lm
                backend when integration lands.
- ``draft``   — External draft model (small Qwen3.5 sibling). Works
                today but uses extra GPU memory.
- ``lookup``  — Prompt-lookup decoding (PLD). No second model, no extra
                memory; effective on prompts where the output overlaps
                with the input (code refactors, doc edits). Implemented
                in ``daydream.pld``.
- ``none``    — No speculation. Baseline.

Per-model capability
--------------------
Each model family declares which methods it *meaningfully supports*.
``recommended_method`` is what Daydream chooses when the user asks for
``auto``. ``available_methods`` is the full set we'll let the user opt
into via ``--speculative``.

For Qwen3.6 the recommended method is ``mtp`` because that's the
upstream guidance, but until the MTP backend lands the recommendation
is *aspirational* — ``auto`` resolves to ``draft`` today and we surface
that fact in `daydream show`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Stable, lowercase string identifiers — used in CLI flags, slash
# commands, model config, JSON payloads, telemetry. Treat as enum.
METHOD_AUTO = "auto"
METHOD_MTP = "mtp"
METHOD_DRAFT = "draft"
METHOD_LOOKUP = "lookup"
METHOD_NONE = "none"

ALL_METHODS: tuple[str, ...] = (
    METHOD_AUTO,
    METHOD_MTP,
    METHOD_DRAFT,
    METHOD_LOOKUP,
    METHOD_NONE,
)

# Recommended num_speculative_tokens per method, per the upstream
# project guidance (vLLM's `qwen3_next_mtp` defaults to 2; classic
# external-draft on hybrid models works best at 2–3; PLD typically
# tops out at 3–5).
_DEFAULT_NUM_SPECULATIVE: dict[str, int] = {
    METHOD_MTP: 2,
    METHOD_DRAFT: 3,
    METHOD_LOOKUP: 4,
}


@dataclass(frozen=True)
class SpeculativeCapability:
    """Per-model speculative-decoding declaration."""

    supported: bool
    recommended: str
    available: tuple[str, ...]
    external_draft_required: bool
    default_num_speculative: int
    notes: str = ""


# Qwen3.6 supports MTP, but we deliberately keep it OPT-IN — the user
# must explicitly type `--speculative mtp` or `/draft mtp`. Reasons:
#   * MTP requires ~16 GB of extra download (the MTPLX-equipped
#     checkpoint bundles its own copy of the trunk weights + MTP head
#     + draft lm_head). We never silently pull that much data.
#   * The MTP head sometimes degrades quality at high temperatures
#     and benefits from a warm-up run — surprising for users who
#     don't know they're on it.
#   * On lower-memory Macs the dual-cache layout (trunk + MTP head)
#     pushes peak GPU memory close to the wired limit.
# `--speculative auto` therefore resolves to `draft` (the external
# Qwen3.5-0.8B draft) on Qwen3.6 — the same conservative default we
# already validated. Users who want MTP get it by asking for it.
QWEN36_CAPABILITY = SpeculativeCapability(
    supported=True,
    # Strict opt-in: --speculative auto resolves to NONE. Users must
    # explicitly pick mtp / draft / lookup to turn on speculation.
    recommended=METHOD_NONE,
    available=(METHOD_MTP, METHOD_DRAFT, METHOD_NONE),
    external_draft_required=False,
    default_num_speculative=_DEFAULT_NUM_SPECULATIVE[METHOD_DRAFT],
    notes=(
        "Qwen3.6 supports model-native MTP (`--speculative mtp`) and "
        "external-draft (`--speculative draft`). Both are opt-in; "
        "`--speculative auto` runs the single model with no extra cost."
    ),
)

QWEN35_CAPABILITY = SpeculativeCapability(
    supported=True,
    recommended=METHOD_NONE,
    available=(METHOD_DRAFT, METHOD_NONE),
    external_draft_required=True,
    default_num_speculative=_DEFAULT_NUM_SPECULATIVE[METHOD_DRAFT],
    notes=(
        "Qwen3.5 supports external-draft (`--speculative draft`). Opt-in only."
    ),
)

UNSUPPORTED_CAPABILITY = SpeculativeCapability(
    supported=False,
    recommended=METHOD_NONE,
    available=(METHOD_NONE, METHOD_LOOKUP),
    external_draft_required=False,
    default_num_speculative=_DEFAULT_NUM_SPECULATIVE[METHOD_LOOKUP],
    notes=(
        "Model-native speculative decoding is not available here. "
        "--speculative lookup (prompt-ngram) can still help on "
        "prompts that share text with the output (code, refactors, "
        "long quotes)."
    ),
)


def _normalize_ref(value: str | None) -> str:
    if not value:
        return ""
    return f"{value} {Path(str(value)).name}".lower()


def capability_for(model_ref: str | None) -> SpeculativeCapability:
    """Return the speculative-decoding capability for a model reference.

    Accepts short names (``qwen3.6:27b``), full HF repo IDs, or local
    paths. Matches by family substring so fine-tunes and re-quants
    inherit the right capability automatically.
    """
    key = _normalize_ref(model_ref)
    if "qwen3.6" in key:
        return QWEN36_CAPABILITY
    if "qwen3.5" in key:
        return QWEN35_CAPABILITY
    return UNSUPPORTED_CAPABILITY


def normalize_method(value: str | None) -> str:
    """Normalize a user-supplied method string. Defaults to ``auto``."""
    if not value:
        return METHOD_AUTO
    text = str(value).strip().lower()
    if text in ("", "default"):
        return METHOD_AUTO
    if text not in ALL_METHODS:
        raise ValueError(
            f"Unknown --speculative method '{value}'. "
            f"Pick one of: {', '.join(ALL_METHODS)}."
        )
    return text


def resolve_method(
    requested: str | None,
    *,
    model_ref: str | None,
    mtp_backend_available: bool = False,
) -> tuple[str, str | None]:
    """Decide which method to actually run.

    Returns ``(effective_method, fallback_reason)``. ``fallback_reason``
    is None when the requested method is what we'll run; otherwise it
    explains why we degraded.

    Crucial design choice: ``METHOD_AUTO`` NEVER resolves to ``mtp``.
    MTP is an explicit opt-in (large download, dual cache, hardware-
    sensitive). The user must type `--speculative mtp` or `/draft mtp`.
    """
    method = normalize_method(requested)
    cap = capability_for(model_ref)

    if method == METHOD_NONE:
        return METHOD_NONE, None

    if method == METHOD_MTP:
        if not mtp_backend_available:
            for fallback in (METHOD_DRAFT, METHOD_LOOKUP, METHOD_NONE):
                if fallback in cap.available or fallback == METHOD_NONE:
                    return fallback, (
                        "MTP backend is not ready (missing mtplx package "
                        f"or MTP-equipped checkpoint). Falling back to "
                        f"--speculative {fallback}."
                    )
        return METHOD_MTP, None

    if method == METHOD_DRAFT:
        if METHOD_DRAFT not in cap.available:
            return METHOD_NONE, (
                "This model family does not have a Daydream-recommended "
                "external draft pairing. Try --speculative lookup."
            )
        return METHOD_DRAFT, None

    if method == METHOD_LOOKUP:
        if METHOD_LOOKUP not in cap.available:
            return METHOD_NONE, (
                "Prompt-lookup decoding is not safe on this model "
                "(non-trimmable hybrid cache)."
            )
        return METHOD_LOOKUP, None

    # METHOD_AUTO: explicit OPT-IN policy for MTP — auto never picks
    # it. Falls through to the model's `recommended` method, which
    # for Qwen3.6 is METHOD_DRAFT (not MTP).
    return cap.recommended, None


def describe_capability(model_ref: str | None) -> list[str]:
    """Return human-readable lines for `daydream show <model>` output."""
    cap = capability_for(model_ref)
    lines: list[str] = []
    if cap.supported:
        lines.append(f"Speculative decoding: [bold green]supported[/]")
        lines.append(f"  Recommended method: [bold]{cap.recommended}[/]")
        available_str = " / ".join(m for m in cap.available if m != METHOD_NONE) or "—"
        lines.append(f"  Available methods : {available_str}")
        lines.append(
            f"  External draft   : "
            f"{'required' if cap.external_draft_required else 'not required'}"
        )
        lines.append(f"  Default draft tokens: {cap.default_num_speculative}")
    else:
        lines.append("Speculative decoding: [dim]not supported (model-native)[/dim]")
        lines.append("  [dim]Try --speculative lookup for ngram-based PLD on this model.[/dim]")
    if cap.notes:
        lines.append(f"  [dim]Notes: {cap.notes}[/dim]")
    return lines


def methods_for(model_ref: str | None) -> tuple[str, ...]:
    """Return the set of methods a model meaningfully supports."""
    return capability_for(model_ref).available


def is_mtp_recommended(model_ref: str | None) -> bool:
    return capability_for(model_ref).recommended == METHOD_MTP


class MTPNotAvailable(RuntimeError):
    """Raised when --speculative mtp is requested but the backend
    isn't wired (Phase 1 reality)."""

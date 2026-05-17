"""Daydream generation backends.

Each backend implements one speculative-decoding (or non-speculative)
path. The dispatcher in `daydream.engine` picks one based on the
resolved `SpeculativeMethod` and the loaded model.

Backends are intentionally thin: they wrap the existing low-level
generation primitives in `daydream.engine` / `daydream.pld` so the
abstraction is a routing layer, not a re-implementation of generation.

Public surface:
    GenerationBackend           — ABC for all backends
    SamplingParams              — request-time parameters
    SpeculativeParams           — speculative-decoding parameters
    BackendCapabilities         — what a backend supports
    backend_for(method, model)  — factory

Phase 2A note: `MTPBackend` is a scaffolded implementation. It detects
an MTPLX-style sidecar checkpoint, exposes capability metadata, and
contains the documented MTP verify-loop skeleton, but the actual
forward pass raises `NotImplementedError` until the sidecar weight
loader is verified on real hardware (Phase 2B).
"""

from __future__ import annotations

from .base import (
    BackendCapabilities,
    GenerationBackend,
    SamplingParams,
    SpeculativeParams,
    backend_for,
)
from .mlx_native import MLXBackend
from .lookup import LookupBackend
from .mtp import MTPBackend, MTPSidecarMissing, detect_mtp_sidecar

__all__ = [
    "BackendCapabilities",
    "GenerationBackend",
    "SamplingParams",
    "SpeculativeParams",
    "backend_for",
    "MLXBackend",
    "LookupBackend",
    "MTPBackend",
    "MTPSidecarMissing",
    "detect_mtp_sidecar",
]

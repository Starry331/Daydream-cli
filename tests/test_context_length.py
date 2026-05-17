"""Unit tests for the /context length helpers."""
from __future__ import annotations

import unittest

from daydream.config import (
    CONTEXT_LENGTH_HARD_CEILING,
    DEFAULT_CONTEXT_LENGTH,
    _normalize_context_length,
    resolve_context_length,
)


class ContextLengthTests(unittest.TestCase):
    def test_normalize_accepts_presets(self) -> None:
        self.assertEqual(_normalize_context_length("auto"), "auto")
        self.assertEqual(_normalize_context_length("DEFAULT"), "auto")
        self.assertEqual(_normalize_context_length(" 4K "), "4k")
        self.assertEqual(_normalize_context_length("32k"), "32k")
        self.assertEqual(_normalize_context_length("128K"), "128k")

    def test_normalize_accepts_custom_int(self) -> None:
        self.assertEqual(_normalize_context_length("65536"), "65536")
        self.assertEqual(_normalize_context_length("12000"), "12000")
        self.assertEqual(_normalize_context_length(20000), "20000")
        # Suffix forms.
        self.assertEqual(_normalize_context_length("50k"), str(50 * 1024))
        # Garbage falls back to default.
        self.assertEqual(_normalize_context_length("garbage"), DEFAULT_CONTEXT_LENGTH)

    def test_hard_ceiling_clamps_huge_values(self) -> None:
        # A user who typed 10M should not be able to lock themselves out.
        clamped = _normalize_context_length("10000000")
        self.assertEqual(int(clamped), CONTEXT_LENGTH_HARD_CEILING)
        # Same via the resolver.
        resolved = resolve_context_length("10000000")
        self.assertEqual(resolved, CONTEXT_LENGTH_HARD_CEILING)

    def test_resolve_auto_returns_none(self) -> None:
        self.assertIsNone(resolve_context_length("auto"))
        self.assertIsNone(resolve_context_length(None))

    def test_resolve_preset_returns_int(self) -> None:
        self.assertEqual(resolve_context_length("32k"), 32 * 1024)
        self.assertEqual(resolve_context_length("4k"), 4 * 1024)

    def test_resolve_clamps_to_model_cap(self) -> None:
        # Even with a generous preset, never exceed the model's
        # max_position_embeddings — otherwise mlx-lm will reject.
        self.assertEqual(
            resolve_context_length("128k", model_max_position_embeddings=8192),
            8192,
        )

    def test_resolve_invalid_returns_none(self) -> None:
        self.assertIsNone(resolve_context_length("garbage"))


if __name__ == "__main__":
    unittest.main()

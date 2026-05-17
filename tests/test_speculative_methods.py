"""Tests for the speculative-method capability table and resolver.

This is Phase 1 surface — the table itself, the resolver, and the
auto-fallback contract. Phase 2 will plug a real MTP backend in
behind the `mtp_backend_available=True` parameter; these tests
already exercise that future-mode path.
"""
from __future__ import annotations

import unittest

from daydream.speculative_methods import (
    METHOD_AUTO,
    METHOD_DRAFT,
    METHOD_LOOKUP,
    METHOD_MTP,
    METHOD_NONE,
    capability_for,
    is_mtp_recommended,
    methods_for,
    normalize_method,
    resolve_method,
)


class CapabilityTableTests(unittest.TestCase):
    def test_qwen36_lists_methods_but_recommends_none(self) -> None:
        # Strict opt-in: --speculative auto never silently turns on
        # speculation. User must explicitly pick mtp / draft / lookup.
        cap = capability_for("mlx-community/Qwen3.6-27B-4bit")
        self.assertTrue(cap.supported)
        self.assertEqual(cap.recommended, METHOD_NONE)
        self.assertIn(METHOD_MTP, cap.available)
        self.assertIn(METHOD_DRAFT, cap.available)
        self.assertFalse(cap.external_draft_required)

    def test_qwen36_35b_a3b_also_recommends_none(self) -> None:
        cap = capability_for("mlx-community/Qwen3.6-35B-A3B-4bit")
        self.assertEqual(cap.recommended, METHOD_NONE)
        self.assertIn(METHOD_MTP, cap.available)

    def test_qwen35_recommends_none(self) -> None:
        cap = capability_for("mlx-community/Qwen3.5-9B-MLX-4bit")
        self.assertEqual(cap.recommended, METHOD_NONE)
        self.assertTrue(cap.external_draft_required)

    def test_unsupported_model(self) -> None:
        cap = capability_for("mlx-community/Llama-3.2-3B-Instruct-4bit")
        self.assertFalse(cap.supported)
        # Lookup is universally available.
        self.assertIn(METHOD_LOOKUP, cap.available)

    def test_methods_for_passthrough(self) -> None:
        self.assertEqual(
            set(methods_for("qwen3.6:27b")),
            set(capability_for("qwen3.6:27b").available),
        )

    def test_mtp_is_never_the_recommended_method(self) -> None:
        # Opt-in policy: no model family lists MTP as the recommended
        # default, even when MTP is supported. Users explicitly request
        # it. Locks the invariant via test.
        self.assertFalse(is_mtp_recommended("qwen3.6:27b"))
        self.assertFalse(is_mtp_recommended("qwen3.5:9b"))
        self.assertFalse(is_mtp_recommended("qwen3:8b"))


class NormalizeMethodTests(unittest.TestCase):
    def test_none_value_defaults_to_auto(self) -> None:
        self.assertEqual(normalize_method(None), METHOD_AUTO)
        self.assertEqual(normalize_method(""), METHOD_AUTO)
        self.assertEqual(normalize_method("  default "), METHOD_AUTO)

    def test_accepts_known_methods_case_insensitive(self) -> None:
        self.assertEqual(normalize_method("MTP"), METHOD_MTP)
        self.assertEqual(normalize_method("Draft"), METHOD_DRAFT)

    def test_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            normalize_method("garbage")


class ResolveMethodTests(unittest.TestCase):
    def test_mtp_without_backend_falls_back_to_draft(self) -> None:
        method, reason = resolve_method(
            "mtp",
            model_ref="qwen3.6:27b",
            mtp_backend_available=False,
        )
        self.assertEqual(method, METHOD_DRAFT)
        self.assertIsNotNone(reason)
        self.assertIn("MTP backend is not ready", reason)

    def test_mtp_with_backend_runs_mtp(self) -> None:
        method, reason = resolve_method(
            "mtp",
            model_ref="qwen3.6:27b",
            mtp_backend_available=True,
        )
        self.assertEqual(method, METHOD_MTP)
        self.assertIsNone(reason)

    def test_auto_on_qwen36_picks_none(self) -> None:
        # Strict opt-in: auto always resolves to NONE — never enables
        # speculation behind the user's back.
        method, reason = resolve_method(
            "auto",
            model_ref="qwen3.6:27b",
            mtp_backend_available=False,
        )
        self.assertEqual(method, METHOD_NONE)
        self.assertIsNone(reason)

    def test_auto_never_picks_speculation_even_when_ready(self) -> None:
        # Even with mtplx fully installed + checkpoint cached, auto
        # MUST stay at NONE. Speculation is always explicit.
        method, _ = resolve_method(
            "auto",
            model_ref="qwen3.6:27b",
            mtp_backend_available=True,
        )
        self.assertEqual(method, METHOD_NONE)

    def test_draft_on_unsupported_model_returns_none(self) -> None:
        method, reason = resolve_method(
            "draft",
            model_ref="mlx-community/Llama-3.2-3B-Instruct-4bit",
            mtp_backend_available=False,
        )
        self.assertEqual(method, METHOD_NONE)
        self.assertIsNotNone(reason)

    def test_lookup_on_qwen36_returns_none(self) -> None:
        # Qwen3.6's hybrid cache can't be trimmed → PLD is unsafe and
        # absent from `available`. Resolver should refuse cleanly.
        method, reason = resolve_method(
            "lookup",
            model_ref="qwen3.6:27b",
            mtp_backend_available=False,
        )
        self.assertEqual(method, METHOD_NONE)
        self.assertIsNotNone(reason)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from daydream.speculative import (
    QWEN35_DRAFT_MODEL,
    QWEN35_NUM_DRAFT_TOKENS,
    QWEN36_DRAFT_MODEL,
    QWEN36_NUM_DRAFT_TOKENS,
    default_draft_for_model,
    default_num_draft_tokens,
    draft_model_for_family,
    is_qwen35_runtime_model,
    is_qwen36_runtime_model,
    is_qwen_hybrid_runtime_model,
    supports_manual_draft,
)


class SpeculativeTests(unittest.TestCase):
    def test_qwen35_detection_matches_supported_refs(self) -> None:
        self.assertTrue(is_qwen35_runtime_model("qwen3.5:9b"))
        self.assertTrue(is_qwen35_runtime_model("mlx-community/Qwen3.5-9B-MLX-4bit"))
        self.assertTrue(is_qwen35_runtime_model("/models/Qwen3.5-9B-MLX-4bit"))

    def test_qwen35_detection_rejects_other_models(self) -> None:
        self.assertFalse(is_qwen35_runtime_model("qwen3:8b"))
        self.assertFalse(is_qwen35_runtime_model("qwen3.6:27b"))
        self.assertFalse(is_qwen35_runtime_model("mlx-community/SmolLM2-135M-Instruct-4bit"))

    def test_qwen36_detection_matches_supported_refs(self) -> None:
        self.assertTrue(is_qwen36_runtime_model("qwen3.6:27b"))
        self.assertTrue(is_qwen36_runtime_model("mlx-community/Qwen3.6-27B-4bit"))
        self.assertTrue(is_qwen36_runtime_model("mlx-community/Qwen3.6-27B-OptiQ-4bit"))
        self.assertTrue(is_qwen36_runtime_model("/models/Qwen3.6-27B-4bit"))

    def test_qwen36_detection_rejects_other_models(self) -> None:
        self.assertFalse(is_qwen36_runtime_model("qwen3.5:9b"))
        self.assertFalse(is_qwen36_runtime_model("qwen3:8b"))

    def test_hybrid_family_detection_covers_both(self) -> None:
        self.assertTrue(is_qwen_hybrid_runtime_model("qwen3.5:9b"))
        self.assertTrue(is_qwen_hybrid_runtime_model("qwen3.6:27b"))
        self.assertFalse(is_qwen_hybrid_runtime_model("qwen3:8b"))

    def test_default_draft_is_none_for_every_family(self) -> None:
        # Strict opt-in policy: NO family auto-enables draft. Users
        # must type `--speculative draft` / `/draft on` explicitly.
        self.assertIsNone(default_draft_for_model("qwen3.6:27b"))
        self.assertIsNone(default_draft_for_model("qwen3.6:35b-a3b"))
        self.assertIsNone(default_draft_for_model("qwen3.5:14b"))
        self.assertIsNone(default_draft_for_model("qwen3:8b"))
        # The num-draft-tokens lookup is still wired (used when the
        # user explicitly opts in via --speculative draft).
        self.assertEqual(default_num_draft_tokens("qwen3.6:27b"), QWEN36_NUM_DRAFT_TOKENS)

    def test_num_draft_tokens_per_family(self) -> None:
        self.assertEqual(default_num_draft_tokens("qwen3.5:14b"), QWEN35_NUM_DRAFT_TOKENS)
        self.assertEqual(default_num_draft_tokens("qwen3.6:27b"), QWEN36_NUM_DRAFT_TOKENS)
        # The Qwen3.6 default is intentionally higher than Qwen3.5's
        # because the speculative path has more headroom there.
        self.assertGreater(QWEN36_NUM_DRAFT_TOKENS, QWEN35_NUM_DRAFT_TOKENS)

    def test_manual_draft_supported_for_qwen35_and_qwen36(self) -> None:
        self.assertEqual(draft_model_for_family("qwen3.5:14b"), QWEN35_DRAFT_MODEL)
        self.assertEqual(default_num_draft_tokens("qwen3.5:14b"), QWEN35_NUM_DRAFT_TOKENS)
        self.assertEqual(draft_model_for_family("qwen3.6:27b"), QWEN36_DRAFT_MODEL)
        self.assertTrue(supports_manual_draft("qwen3.5:14b"))
        self.assertTrue(supports_manual_draft("qwen3.6:27b"))
        self.assertFalse(supports_manual_draft("qwen3:8b"))


if __name__ == "__main__":
    unittest.main()

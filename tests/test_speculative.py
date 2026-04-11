from __future__ import annotations

import unittest

from daydream.speculative import (
    QWEN35_DRAFT_MODEL,
    default_draft_for_model,
    default_num_draft_tokens,
    is_qwen35_runtime_model,
    supports_manual_draft,
)


class SpeculativeTests(unittest.TestCase):
    def test_qwen35_detection_matches_supported_refs(self) -> None:
        self.assertTrue(is_qwen35_runtime_model("qwen3.5:9b"))
        self.assertTrue(is_qwen35_runtime_model("mlx-community/Qwen3.5-9B-MLX-4bit"))
        self.assertTrue(is_qwen35_runtime_model("/models/Qwen3.5-9B-MLX-4bit"))

    def test_qwen35_detection_rejects_other_models(self) -> None:
        self.assertFalse(is_qwen35_runtime_model("qwen3:8b"))
        self.assertFalse(is_qwen35_runtime_model("mlx-community/SmolLM2-135M-Instruct-4bit"))

    def test_default_draft_mapping_is_qwen35_only(self) -> None:
        self.assertEqual(default_draft_for_model("qwen3.5:14b"), QWEN35_DRAFT_MODEL)
        self.assertEqual(default_num_draft_tokens("qwen3.5:14b"), 6)
        self.assertIsNone(default_draft_for_model("qwen3:8b"))
        self.assertFalse(supports_manual_draft("qwen3:8b"))


if __name__ == "__main__":
    unittest.main()

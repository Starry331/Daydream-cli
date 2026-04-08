from __future__ import annotations

import unittest

from daydream.utils import render_daydreaming_text, render_status_footer, render_title_text


class DaydreamingTextTests(unittest.TestCase):
    def test_render_daydreaming_text_contains_label(self) -> None:
        rendered = render_daydreaming_text(2)
        self.assertIn("Daydreaming", rendered.plain)

    def test_render_title_text_contains_label(self) -> None:
        rendered = render_title_text("Downloading model", 3)
        self.assertIn("Daydream CLI", rendered)
        self.assertIn("Downloading model", rendered)

    def test_render_status_footer_contains_model_and_rate(self) -> None:
        rendered = render_status_footer("qwen3:8b", tokens_per_second=42.5, phase="prefill")
        self.assertIn("qwen3:8b", rendered.plain)
        self.assertIn("42.5 tok/s", rendered.plain)
        self.assertIn("prefill", rendered.plain)


if __name__ == "__main__":
    unittest.main()

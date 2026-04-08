from __future__ import annotations

import unittest

from daydream.utils import render_daydreaming_text, render_title_text


class DaydreamingTextTests(unittest.TestCase):
    def test_render_daydreaming_text_contains_label(self) -> None:
        rendered = render_daydreaming_text(2)
        self.assertIn("Daydreaming", rendered.plain)

    def test_render_title_text_contains_label(self) -> None:
        rendered = render_title_text("Downloading model", 3)
        self.assertIn("Daydream CLI", rendered)
        self.assertIn("Downloading model", rendered)


if __name__ == "__main__":
    unittest.main()

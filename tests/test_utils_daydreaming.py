from __future__ import annotations

import unittest

from daydream.utils import render_daydreaming_text


class DaydreamingTextTests(unittest.TestCase):
    def test_render_daydreaming_text_contains_label(self) -> None:
        rendered = render_daydreaming_text(2)
        self.assertIn("Daydreaming", rendered.plain)


if __name__ == "__main__":
    unittest.main()

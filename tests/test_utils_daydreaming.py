from __future__ import annotations

import unittest

from daydream.utils import (
    _strip_ansi,
    build_effort_menu_lines,
    build_input_box_lines,
    render_daydreaming_text,
    render_status_footer,
    render_title_text,
)


class DaydreamingTextTests(unittest.TestCase):
    def test_render_daydreaming_text_contains_label(self) -> None:
        rendered = render_daydreaming_text(2)
        self.assertIn("Daydreaming", rendered.plain)

    def test_render_daydreaming_text_cycles_z_cluster(self) -> None:
        counts = []
        for frame in (0, 12, 24, 36):
            rendered = render_daydreaming_text(frame)
            counts.append(rendered.plain.split("Daydreaming", 1)[0].count("Z"))
        self.assertEqual(counts, [1, 2, 3, 0])

    def test_render_title_text_contains_label(self) -> None:
        rendered = render_title_text("Downloading model", 3)
        self.assertIn("Daydream CLI", rendered)
        self.assertIn("Downloading model", rendered)

    def test_render_title_text_accepts_custom_frames(self) -> None:
        rendered = render_title_text("Daydreaming", 1, frames=("Z", "ZZ"))
        self.assertIn("ZZ Daydream CLI", rendered)

    def test_render_status_footer_contains_model_and_rate(self) -> None:
        rendered = render_status_footer("qwen3:8b", tokens_per_second=42.5, phase="prefill")
        self.assertIn("qwen3:8b", rendered.plain)
        self.assertIn("42.5 tok/s", rendered.plain)
        self.assertIn("prefill", rendered.plain)

    def test_build_input_box_lines_have_consistent_width(self) -> None:
        lines = build_input_box_lines(
            ["/"],
            command_rows=[("/effort", "adjust reasoning depth")],
            selected_command="/effort",
        )
        widths = {len(_strip_ansi(line)) for line in lines}
        self.assertEqual(len(widths), 1)
        self.assertIn("› /effort", _strip_ansi("".join(lines)))
        self.assertTrue(lines[-1].startswith("_"))
        self.assertNotIn("Send", "".join(lines))

    def test_build_effort_menu_lines_have_consistent_width(self) -> None:
        lines = build_effort_menu_lines("default", "long", supported=True)
        widths = {len(_strip_ansi(line)) for line in lines}
        self.assertEqual(len(widths), 1)


if __name__ == "__main__":
    unittest.main()

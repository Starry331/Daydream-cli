from __future__ import annotations

import io
import os
import unittest

from rich.console import Console

from daydream.utils import (
    _strip_ansi,
    BottomTerminalRenderer,
    _DREAM_WORDS,
    REFLECTING_LABEL,
    build_effort_menu_lines,
    build_input_box_lines,
    choose_dream_word,
    render_sleep_phase_box,
    render_daydreaming_text,
    render_status_footer,
    render_title_text,
)


class DaydreamingTextTests(unittest.TestCase):
    def test_render_daydreaming_text_contains_label(self) -> None:
        rendered = render_daydreaming_text(2)
        self.assertTrue(any(word in rendered.plain for word in _DREAM_WORDS))

    def test_render_daydreaming_text_cycles_z_cluster(self) -> None:
        counts = []
        for frame in (0, 12, 24, 36):
            rendered = render_daydreaming_text(frame, label="Imagining")
            counts.append(rendered.plain.split("Imagining", 1)[0].count("Z"))
        self.assertEqual(counts, [1, 2, 3, 0])

    def test_choose_dream_word_returns_supported_values(self) -> None:
        for _ in range(20):
            self.assertIn(choose_dream_word(), _DREAM_WORDS)

    def test_render_daydreaming_text_keeps_dots_when_label_changes(self) -> None:
        rendered = render_daydreaming_text(6, label="Rêvant")
        self.assertIn("Rêvant", rendered.plain)
        self.assertTrue(rendered.plain.endswith("···"))

    def test_render_sleep_phase_box_uses_reflecting_label(self) -> None:
        stream = io.StringIO()
        console = Console(file=stream, record=True, force_terminal=False, width=80)
        console.print(render_sleep_phase_box("n3", "line one", 3))
        exported = console.export_text()
        self.assertIn(REFLECTING_LABEL, exported)

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

    def test_bottom_terminal_renderer_clears_only_bottom_region_on_resize(self) -> None:
        stream = io.StringIO()

        class Renderer(BottomTerminalRenderer):
            def __init__(self, stream):
                super().__init__(stream, clear_on_finish=False)
                self.rows = 24
                self.cols = 96

            def _terminal_size(self):
                return os.terminal_size((self.cols, self.rows))

        renderer = Renderer(stream)
        renderer.render(["first", "second", "third"])
        renderer.cols = 60
        renderer.rows = 18
        renderer.render(["first", "second", "third"])
        renderer.finish()

        output = stream.getvalue()
        self.assertIn("\x1b[J", output)
        self.assertNotIn("\x1b[2J", output)
        self.assertIn("\x1b[22;1H", output)
        self.assertIn("\x1b[16;1H", output)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import io
import os
import unittest
from unittest import mock

from rich.console import Console

from daydream.utils import (
    _strip_ansi,
    BottomTerminalRenderer,
    CLI_PAGE_MODES,
    ConversationStatus,
    InlineFlowRenderer,
    _DREAM_WORDS,
    REFLECTING_LABEL,
    build_cli_page_menu_lines,
    build_effort_menu_lines,
    build_input_box_lines,
    choose_dream_word,
    render_sleep_phase_box,
    render_daydreaming_text,
    render_reasoning_box,
    render_status_footer,
    render_title_text,
)


def measure_lines(renderable) -> int:
    stream = io.StringIO()
    console = Console(file=stream, record=True, force_terminal=False, width=80)
    console.print(renderable)
    return len(console.export_text().splitlines())


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

    def test_compact_reasoning_box_uses_fewer_lines(self) -> None:
        loose = measure_lines(render_reasoning_box("line1\nline2\nline3\nline4", 0, compact=False))
        tight = measure_lines(render_reasoning_box("line1\nline2\nline3\nline4", 0, compact=True))
        self.assertLess(tight, loose)

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

    def test_tight_input_box_lines_use_fewer_rows(self) -> None:
        loose = build_input_box_lines(
            ["/"],
            command_rows=[("/effort", "adjust reasoning depth")],
            selected_command="/effort",
            cli_page_mode="loose",
        )
        tight = build_input_box_lines(
            ["/"],
            command_rows=[("/effort", "adjust reasoning depth")],
            selected_command="/effort",
            cli_page_mode="tight",
        )
        self.assertLess(len(tight), len(loose))

    def test_build_effort_menu_lines_have_consistent_width(self) -> None:
        lines = build_effort_menu_lines("default", "long", supported=True)
        widths = {len(_strip_ansi(line)) for line in lines}
        self.assertEqual(len(widths), 1)

    def test_tight_effort_menu_lines_use_fewer_rows(self) -> None:
        loose = build_effort_menu_lines("default", "long", supported=True, cli_page_mode="loose")
        tight = build_effort_menu_lines("default", "long", supported=True, cli_page_mode="tight")
        self.assertLess(len(tight), len(loose))

    def test_build_cli_page_menu_lines_have_consistent_width(self) -> None:
        lines = build_cli_page_menu_lines("loose", "tight")
        widths = {len(_strip_ansi(line)) for line in lines}
        self.assertEqual(len(widths), 1)
        plain = "".join(_strip_ansi(line) for line in lines)
        for option in CLI_PAGE_MODES:
            self.assertIn(option, plain)

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
        self.assertIn("\x1b[18;1H\n\n\n\n\n\n", output)

    def test_bottom_terminal_renderer_scrolls_cleanly_when_overlay_grows(self) -> None:
        stream = io.StringIO()

        class Renderer(BottomTerminalRenderer):
            def __init__(self, stream):
                super().__init__(stream, clear_on_finish=False)
                self.rows = 24
                self.cols = 96

            def _terminal_size(self):
                return os.terminal_size((self.cols, self.rows))

        renderer = Renderer(stream)
        renderer.render(["line one", "line two", "line three"])
        before_growth = len(stream.getvalue())
        renderer.render([
            "line one",
            "line two",
            "line three",
            "line four",
            "line five",
            "line six",
        ])

        growth_output = stream.getvalue()[before_growth:]
        self.assertIn("\x1b[22;1H\x1b[J", growth_output)
        self.assertIn("\x1b[24;1H\n\n\n", growth_output)
        self.assertLess(
            growth_output.index("\x1b[22;1H\x1b[J"),
            growth_output.index("\x1b[24;1H\n\n\n"),
        )

    def test_bottom_terminal_renderer_can_collapse_reserved_rows_on_finish(self) -> None:
        stream = io.StringIO()

        class Renderer(BottomTerminalRenderer):
            def __init__(self, stream):
                super().__init__(stream, clear_on_finish=True, collapse_on_finish=True)
                self.rows = 24
                self.cols = 96

            def _terminal_size(self):
                return os.terminal_size((self.cols, self.rows))

        renderer = Renderer(stream)
        renderer.render(["one", "two", "three"])
        renderer.finish()

        output = stream.getvalue()
        self.assertIn("\x1b[22;1H\x1b[J", output)
        self.assertIn("\x1b[3M", output)

    def test_inline_flow_renderer_uses_insert_delete_lines_for_reflow(self) -> None:
        stream = io.StringIO()

        class Renderer(InlineFlowRenderer):
            def __init__(self, stream):
                super().__init__(stream, clear_on_finish=True)
                self.rows = 24
                self.cols = 96

            def _terminal_size(self):
                return os.terminal_size((self.cols, self.rows))

        renderer = Renderer(stream)
        renderer.render(["one", "two", "three"])
        renderer.render(["one", "two", "three", "four", "five"])
        renderer.render(["one", "two"])
        renderer.finish()

        output = stream.getvalue()
        self.assertIn("\x1b[3L", output)
        self.assertIn("\x1b[2L", output)
        self.assertIn("\x1b[3M", output)
        self.assertIn("\x1b[2M", output)
        self.assertNotIn("\n\n\n\n\n\n\n\n\n", output)

    def test_conversation_status_stop_disables_future_overlay_renders(self) -> None:
        stream = io.StringIO()
        stream.isatty = lambda: True
        console = Console(file=stream, force_terminal=True, color_system="truecolor")
        status = ConversationStatus(console, "qwen3.5-9b")
        status.start()
        status.start_reasoning()
        status.append_reasoning("thinking")
        status.stop()
        snapshot = stream.getvalue()
        status.update(tokens_per_second=42.0)
        status.append_output("visible text")
        self.assertEqual(stream.getvalue(), snapshot)

    def test_conversation_status_tight_mode_uses_bottom_renderer(self) -> None:
        stream = io.StringIO()
        stream.isatty = lambda: True
        console = Console(file=stream, force_terminal=True, color_system="truecolor")
        status = ConversationStatus(console, "qwen3.5-9b", cli_page_mode="tight")

        with mock.patch("daydream.utils.BottomTerminalRenderer") as bottom_cls, mock.patch(
            "daydream.utils.InlineFlowRenderer"
        ) as inline_cls:
            renderer = mock.Mock()
            bottom_cls.return_value = renderer
            status.start()
            status.stop()

        bottom_cls.assert_called_once()
        self.assertTrue(bottom_cls.call_args.kwargs["collapse_on_finish"])
        inline_cls.assert_not_called()

    def test_conversation_status_append_output_trims_live_overlay_tail(self) -> None:
        stream = io.StringIO()
        console = Console(file=stream, force_terminal=False, width=40)
        status = ConversationStatus(console, "qwen3.5-9b")

        with mock.patch(
            "daydream.utils.shutil.get_terminal_size",
            return_value=os.terminal_size((40, 12)),
        ):
            status.append_output("A" * 4000 + "\nTAIL")

        self.assertLess(len(status._output), 4000)
        self.assertTrue(status._output.endswith("TAIL"))

    def test_conversation_status_append_output_keeps_recent_visual_rows(self) -> None:
        stream = io.StringIO()
        console = Console(file=stream, force_terminal=False, width=40)
        status = ConversationStatus(console, "qwen3.5-9b")
        text = "\n".join(f"line {i} " + ("x" * 80) for i in range(60))

        with mock.patch(
            "daydream.utils.shutil.get_terminal_size",
            return_value=os.terminal_size((40, 12)),
        ):
            status.append_output(text)

        self.assertNotIn("line 0", status._output)
        self.assertNotIn("line 40", status._output)
        self.assertIn("line 58", status._output)
        self.assertIn("line 59", status._output)

    def test_conversation_status_coalesces_rapid_output_renders(self) -> None:
        stream = io.StringIO()
        console = Console(file=stream, force_terminal=False, width=40)
        status = ConversationStatus(console, "qwen3.5-9b")
        status._renderer = mock.Mock()
        status._stopped = False
        status._last_render_at = 10.0

        with mock.patch("daydream.utils.time.monotonic", return_value=10.01):
            status.append_output("a")
        with mock.patch("daydream.utils.time.monotonic", return_value=10.03):
            status.append_output("b")
        self.assertEqual(status._renderer.render.call_count, 0)

        with mock.patch("daydream.utils.time.monotonic", return_value=10.08):
            status.append_output("c")
        self.assertEqual(status._renderer.render.call_count, 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import io
import os
import time
import unittest
from contextlib import contextmanager
from unittest import mock

from rich.console import Console

from daydream.chat import (
    _InlineTerminalRenderer,
    _ReasoningParser,
    _build_request_messages,
    _collect_multiline_message,
    _current_command_selection,
    _drain_pending_escape,
    _effort_chat_template_kwargs,
    _effort_system_prompt,
    _extract_visible_text,
    _is_down_key,
    _is_up_key,
    _merge_escape_key,
    _matching_slash_commands,
    _model_supports_effort,
    _normalize_effort,
    _read_key,
    _read_live_boxed_message,
    _select_effort,
    run_oneshot,
)


class ChatTests(unittest.TestCase):
    def test_collect_multiline_with_triple_quote_sentinel(self) -> None:
        lines = iter(["line one", "line two", '"""'])
        result = _collect_multiline_message('"""', lambda _: next(lines))
        self.assertEqual(result, "line one\nline two")

    def test_collect_multiline_with_backslash_continuation(self) -> None:
        lines = iter(["second line\\", "third line"])
        result = _collect_multiline_message("first line\\", lambda _: next(lines))
        self.assertEqual(result, "first line\nsecond line\nthird line")

    def test_extract_visible_text_hides_think_blocks(self) -> None:
        visible, reasoning, in_reasoning = _extract_visible_text("<think>hidden</think>Hello")
        self.assertEqual(visible, "Hello")
        self.assertEqual(reasoning, "hidden")
        self.assertFalse(in_reasoning)

    def test_reasoning_parser_returns_only_visible_delta(self) -> None:
        parser = _ReasoningParser()
        delta, reasoning_delta, closed = parser.feed("<think>hidden")
        self.assertEqual(delta, "")
        self.assertEqual(reasoning_delta, "hidden")
        self.assertFalse(closed)

        delta, reasoning_delta, closed = parser.feed("</think>Hello")
        self.assertEqual(delta, "Hello")
        self.assertEqual(reasoning_delta, "")
        self.assertTrue(closed)

    def test_reasoning_parser_detects_plain_thinking_prefix(self) -> None:
        parser = _ReasoningParser()
        delta, reasoning_delta, closed = parser.feed("Here's a thinking process:\n1. Analyze.\n")
        self.assertEqual(delta, "")
        self.assertIn("thinking process", reasoning_delta.lower())
        self.assertFalse(closed)

    def test_reasoning_parser_detects_chunked_chinese_reasoning_prefix(self) -> None:
        parser = _ReasoningParser()

        delta, reasoning_delta, closed = parser.feed("好的，用户发来的是/think，")
        self.assertEqual(delta, "")
        self.assertIn("好的，用户", reasoning_delta)
        self.assertFalse(closed)

        delta, reasoning_delta, closed = parser.feed("看起来他们可能想让我开启某种特殊模式。\n")
        self.assertEqual(delta, "")
        self.assertIn("看起来他们可能", reasoning_delta)
        self.assertFalse(closed)

        delta, reasoning_delta, closed = parser.feed("总结：用户可能误用了命令。\n")
        self.assertEqual(delta, "")
        self.assertIn("总结：用户", reasoning_delta)
        self.assertFalse(closed)

        delta, reasoning_delta, closed = parser.feed("你好")
        self.assertEqual(delta, "你好")
        self.assertEqual(reasoning_delta, "")
        self.assertTrue(closed)

    def test_matching_slash_commands_filters_by_prefix(self) -> None:
        matches = _matching_slash_commands("/e")
        self.assertEqual(matches[0][0], "/effort")
        self.assertTrue(any(name == "/help" for name, _ in _matching_slash_commands("/")))

    def test_current_command_selection_defaults_and_preserves(self) -> None:
        matches, selected = _current_command_selection("/", None, multiline=False)
        self.assertEqual(selected, "/effort")
        self.assertTrue(matches)

        matches, selected = _current_command_selection("/e", "/effort", multiline=False)
        self.assertEqual(selected, "/effort")
        self.assertEqual(matches[0][0], "/effort")

    def test_read_key_supports_ss3_arrow_sequences(self) -> None:
        stdin = mock.Mock()
        stdin.read.side_effect = ["\x1b", "O", "A"]
        stdin.fileno.return_value = 0
        with mock.patch("daydream.chat.sys.stdin", stdin), mock.patch(
            "daydream.chat.select.select",
            side_effect=[([0], [], []), ([0], [], []), ([], [], [])],
        ):
            self.assertEqual(_read_key(), "\x1bOA")

    def test_read_key_waits_for_delayed_csi_sequence(self) -> None:
        stdin = mock.Mock()
        stdin.read.side_effect = ["\x1b", "[", "B"]
        stdin.fileno.return_value = 0
        with mock.patch("daydream.chat.sys.stdin", stdin), mock.patch(
            "daydream.chat.select.select",
            side_effect=[([0], [], []), ([0], [], []), ([], [], [])],
        ):
            self.assertEqual(_read_key(), "\x1b[B")

    def test_slash_menu_handles_fragmented_down_arrow_without_leaking_b(self) -> None:
        @contextmanager
        def fake_raw():
            yield

        class FakeRenderer:
            def __init__(self, *_args, **_kwargs):
                self.lines = []

            def render(self, lines):
                self.lines = list(lines)

            def wait_for_input(self, _lines):
                return

            def finish(self):
                return

        with mock.patch("daydream.chat._raw_stdin", fake_raw), \
            mock.patch("daydream.chat._InlineTerminalRenderer", FakeRenderer), \
            mock.patch(
                "daydream.chat._read_key",
                side_effect=["/", "\x1b", "[", "B", "\r"],
            ):
            result = _read_live_boxed_message()

        self.assertEqual(result, "/help")

    def test_effort_menu_handles_fragmented_down_arrow_without_leaking_b(self) -> None:
        @contextmanager
        def fake_raw():
            yield

        class FakeRenderer:
            def __init__(self, *_args, **_kwargs):
                pass

            def render(self, _lines):
                return

            def wait_for_input(self, _lines):
                return

            def finish(self):
                return

        with mock.patch("daydream.chat._raw_stdin", fake_raw), \
            mock.patch("daydream.chat._InlineTerminalRenderer", FakeRenderer), \
            mock.patch("daydream.chat.sys.stdin.isatty", return_value=True), \
            mock.patch("daydream.chat.err_console", mock.Mock(is_terminal=True)), \
            mock.patch(
                "daydream.chat._read_key",
                side_effect=["\x1b", "[", "B", "\r"],
            ):
            result = _select_effort("default", supported=True)

        self.assertEqual(result, "long")

    def test_arrow_key_helpers_accept_modified_sequences(self) -> None:
        self.assertTrue(_is_up_key("\x1bOA"))
        self.assertTrue(_is_up_key("\x1b[1;2A"))
        self.assertTrue(_is_down_key("\x1b[B"))
        self.assertTrue(_is_down_key("\x1b[1;5B"))

    def test_pending_escape_sequence_merges_fragmented_arrow(self) -> None:
        key, pending, started = _merge_escape_key("\x1b", "", None)
        self.assertIsNone(key)
        self.assertEqual(pending, "\x1b")

        key, pending, started = _merge_escape_key("[", pending, started)
        self.assertIsNone(key)
        self.assertEqual(pending, "\x1b[")

        key, pending, started = _merge_escape_key("B", pending, started)
        self.assertEqual(key, "\x1b[B")
        self.assertEqual(pending, "")
        self.assertIsNone(started)

    def test_pending_escape_drains_to_escape_after_timeout(self) -> None:
        key, pending, started = _drain_pending_escape("\x1b", time.monotonic() - 0.4)
        self.assertEqual(key, "\x1b")
        self.assertEqual(pending, "")
        self.assertIsNone(started)

    def test_effort_helpers_only_emit_prompts_for_supported_models(self) -> None:
        self.assertEqual(_normalize_effort("LONG"), "long")
        self.assertIsNone(_normalize_effort("medium"))
        self.assertTrue(_model_supports_effort("mlx-community/Qwen3-8B-4bit"))
        self.assertFalse(_model_supports_effort("mlx-community/SmolLM2-360M-Instruct-4bit"))
        self.assertIsNotNone(_effort_system_prompt("short", "mlx-community/Qwen3-8B-4bit"))
        self.assertIsNone(_effort_system_prompt("short", "mlx-community/SmolLM2-360M-Instruct-4bit"))
        tokenizer = mock.Mock(has_thinking=True)
        self.assertEqual(_effort_chat_template_kwargs("instant", tokenizer), {"enable_thinking": False})
        self.assertEqual(_effort_chat_template_kwargs("long", tokenizer), {"enable_thinking": True})
        self.assertEqual(_effort_chat_template_kwargs("default", tokenizer), {})

    def test_build_request_messages_includes_effort_system_prompt(self) -> None:
        request = _build_request_messages(
            [{"role": "user", "content": "hello"}],
            system_prompt="base",
            effort="long",
            model_name="mlx-community/Qwen3-8B-4bit",
        )
        self.assertEqual(request[0]["content"], "base")
        self.assertEqual(request[1]["role"], "system")
        self.assertIn("Reasoning effort: long", request[1]["content"])
        self.assertEqual(request[2]["content"], "hello")

    def test_inline_terminal_renderer_tracks_bottom_rows_across_resize(self) -> None:
        stream = io.StringIO()

        class Renderer(_InlineTerminalRenderer):
            def __init__(self, stream):
                super().__init__(stream)
                self.rows = 24
                self.cols = 96

            def _terminal_size(self):
                return os.terminal_size((self.cols, self.rows))

        renderer = Renderer(stream)
        renderer.render(["top", "mid", "bot"])
        renderer.cols = 72
        renderer.rows = 30
        renderer.render(["top", "mid", "bot"])
        renderer.finish()

        output = stream.getvalue()
        self.assertIn("\x1b[J", output)
        self.assertNotIn("\x1b[2J", output)
        self.assertIn("\x1b[?7l", output)
        self.assertIn("\x1b[?7h", output)
        self.assertIn("\x1b[22;1H", output)
        self.assertIn("\x1b[28;1H", output)

    def test_run_oneshot_resolves_before_preparing_load(self) -> None:
        output = io.StringIO()

        with mock.patch("daydream.chat.err_console", Console(file=output, force_terminal=False, color_system=None)), \
            mock.patch("daydream.chat.ensure_runtime_model", return_value="mlx-community/Foo-4bit") as ensure_model, \
            mock.patch("daydream.chat.engine.load_model", return_value=("model", "tokenizer")) as load_model, \
            mock.patch("daydream.chat._stream_response", return_value=("hello", None)):
            run_oneshot("hf.co/mlx-community/Foo-4bit", prompt="hello")

        ensure_model.assert_called_once_with(
            "hf.co/mlx-community/Foo-4bit",
            auto_pull=True,
            register_alias=True,
        )
        load_model.assert_called_once_with("mlx-community/Foo-4bit", ensure_available=False)


if __name__ == "__main__":
    unittest.main()

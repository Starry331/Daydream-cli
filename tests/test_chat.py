from __future__ import annotations

import io
import unittest
from unittest import mock

from rich.console import Console

from daydream.chat import (
    _InlineTerminalRenderer,
    _ReasoningParser,
    _build_request_messages,
    _collect_multiline_message,
    _effort_system_prompt,
    _extract_visible_text,
    _matching_slash_commands,
    _model_supports_effort,
    _normalize_effort,
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

    def test_effort_helpers_only_emit_prompts_for_supported_models(self) -> None:
        self.assertEqual(_normalize_effort("LONG"), "long")
        self.assertIsNone(_normalize_effort("medium"))
        self.assertTrue(_model_supports_effort("mlx-community/Qwen3-8B-4bit"))
        self.assertFalse(_model_supports_effort("mlx-community/SmolLM2-360M-Instruct-4bit"))
        self.assertIsNotNone(_effort_system_prompt("short", "mlx-community/Qwen3-8B-4bit"))
        self.assertIsNone(_effort_system_prompt("short", "mlx-community/SmolLM2-360M-Instruct-4bit"))

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

            def _terminal_rows(self) -> int:
                return self.rows

        renderer = Renderer(stream)
        renderer.render(["top", "mid", "bot"])
        renderer.rows = 30
        renderer.render(["top", "mid", "bot"])

        output = stream.getvalue()
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

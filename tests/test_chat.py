from __future__ import annotations

import io
import unittest
from unittest import mock

from rich.console import Console

from daydream.chat import _ReasoningParser, _collect_multiline_message, _extract_visible_text, run_oneshot


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
        visible, in_reasoning = _extract_visible_text("<think>hidden</think>Hello")
        self.assertEqual(visible, "Hello")
        self.assertFalse(in_reasoning)

    def test_reasoning_parser_returns_only_visible_delta(self) -> None:
        parser = _ReasoningParser()
        delta, closed = parser.feed("<think>hidden")
        self.assertEqual(delta, "")
        self.assertFalse(closed)

        delta, closed = parser.feed("</think>Hello")
        self.assertEqual(delta, "Hello")
        self.assertTrue(closed)

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

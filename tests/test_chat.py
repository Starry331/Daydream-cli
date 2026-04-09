from __future__ import annotations

import io
import os
import time
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

from rich.console import Console

from daydream.chat import (
    _InlineTerminalRenderer,
    _ReasoningParser,
    _STATUS_OVERLAY_RESERVE_LINES,
    _build_request_messages,
    _build_session_memory_prompt,
    _collect_multiline_message,
    _confirm_memory_import,
    _confirm_session_delete,
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
    _select_session_action,
    run_chat,
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

    def test_reasoning_parser_hides_close_tag_after_implicit_reasoning(self) -> None:
        parser = _ReasoningParser()

        delta, reasoning_delta, closed = parser.feed("Here's a thinking process:\n1. Analyze.\n")
        self.assertEqual(delta, "")
        self.assertTrue(reasoning_delta)
        self.assertFalse(closed)

        delta, reasoning_delta, closed = parser.feed("</think>\nHello")
        self.assertEqual(delta, "Hello")
        self.assertEqual(reasoning_delta, "")
        self.assertTrue(closed)

    def test_reasoning_parser_hides_emphasized_reasoning_labels(self) -> None:
        parser = _ReasoningParser()

        delta, reasoning_delta, closed = parser.feed(
            "Here's a thinking process that leads to the suggested response:\n"
        )
        self.assertEqual(delta, "")
        self.assertTrue(reasoning_delta)
        self.assertFalse(closed)

        delta, reasoning_delta, closed = parser.feed(
            "*Simpler:* Hello! How can I help you?\n"
            "*Let's make it friendly:* Hello there! How can I assist you today?\n"
            "*Final Choice:* Hello! How can I help you today? (Classic, effective).\n"
            "*Wait, I should check if there are any specific constraints.* No constraints.\n"
            "*Okay, let's respond.*\n"
        )
        self.assertEqual(delta, "")
        self.assertTrue(reasoning_delta)
        self.assertFalse(closed)

        delta, reasoning_delta, closed = parser.feed("\nHello! How can I help you today?")
        self.assertEqual(delta, "Hello! How can I help you today?")
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
        stdin.fileno.return_value = 0
        with mock.patch("daydream.chat.sys.stdin", stdin), mock.patch(
            "daydream.chat.os.read",
            side_effect=[b"\x1b", b"O", b"A"],
        ), mock.patch(
            "daydream.chat.select.select",
            side_effect=[([0], [], []), ([0], [], []), ([], [], [])],
        ):
            self.assertEqual(_read_key(), "\x1bOA")

    def test_read_key_waits_for_delayed_csi_sequence(self) -> None:
        stdin = mock.Mock()
        stdin.fileno.return_value = 0
        with mock.patch("daydream.chat.sys.stdin", stdin), mock.patch(
            "daydream.chat.os.read",
            side_effect=[b"\x1b", b"[", b"B"],
        ), mock.patch(
            "daydream.chat.select.select",
            side_effect=[([0], [], []), ([0], [], []), ([], [], [])],
        ):
            self.assertEqual(_read_key(), "\x1b[B")

    def test_read_key_decodes_utf8_multibyte_input(self) -> None:
        stdin = mock.Mock()
        stdin.fileno.return_value = 0
        with mock.patch("daydream.chat.sys.stdin", stdin), mock.patch(
            "daydream.chat.os.read",
            side_effect=[b"\xe4", b"\xbd\xa0"],
        ), mock.patch(
            "daydream.chat.select.select",
            side_effect=[([0], [], [])],
        ):
            self.assertEqual(_read_key(), "你")

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
        self.assertEqual(len(request), 2)
        self.assertEqual(request[0]["role"], "system")
        self.assertIn("base", request[0]["content"])
        self.assertIn("Reasoning effort: long", request[0]["content"])
        self.assertEqual(request[1]["content"], "hello")

    def test_build_request_messages_includes_session_memory_prompt(self) -> None:
        from daydream.storage import Memory

        request = _build_request_messages(
            [{"role": "user", "content": "hello"}],
            system_prompt="base",
            effort="default",
            model_name="mlx-community/Qwen3-8B-4bit",
            session_memories=[
                Memory(
                    content="User prefers terse answers.",
                    category="preference",
                    importance=0.9,
                    source_phase="reming",
                )
            ],
        )
        self.assertEqual(len(request), 2)
        self.assertIn("base", request[0]["content"])
        self.assertIn("Session memory for this persistent chat only.", request[0]["content"])
        self.assertIn("User prefers terse answers.", request[0]["content"])
        self.assertEqual(request[1]["content"], "hello")

    def test_build_session_memory_prompt_deduplicates_and_limits(self) -> None:
        from daydream.storage import Memory

        prompt = _build_session_memory_prompt([
            Memory(content="User likes Python.", category="fact", importance=0.4, source_phase="rem"),
            Memory(content="User likes Python.", category="fact", importance=0.9, source_phase="rem"),
        ])
        self.assertIsNotNone(prompt)
        self.assertEqual(prompt.count("User likes Python."), 1)

    def test_build_request_messages_merges_memory_and_effort_into_one_system_message(self) -> None:
        from daydream.storage import Memory

        request = _build_request_messages(
            [{"role": "user", "content": "hello"}],
            system_prompt="base system",
            effort="instant",
            model_name="mlx-community/Qwen3-8B-4bit",
            session_memories=[
                Memory(
                    content="User prefers terse answers.",
                    category="preference",
                    importance=0.9,
                    source_phase="reming",
                )
            ],
        )

        self.assertEqual(len(request), 2)
        self.assertEqual(request[0]["role"], "system")
        self.assertIn("base system", request[0]["content"])
        self.assertIn("User prefers terse answers.", request[0]["content"])
        self.assertIn("Reasoning effort: instant", request[0]["content"])

    def test_confirm_memory_import_defaults_yes(self) -> None:
        with mock.patch("daydream.chat.err_console.input", return_value=""):
            self.assertTrue(_confirm_memory_import([mock.Mock()]))

    def test_confirm_memory_import_respects_no(self) -> None:
        with mock.patch("daydream.chat.err_console.input", return_value="n"):
            self.assertFalse(_confirm_memory_import([mock.Mock()]))

    def test_confirm_session_delete_requires_explicit_choice(self) -> None:
        session = mock.Mock(title="session one")
        with mock.patch("daydream.chat.sys.stdin.isatty", return_value=True), \
            mock.patch("daydream.chat.err_console.input", side_effect=["", "n"]):
            self.assertFalse(_confirm_session_delete(session))

    def test_confirm_session_delete_accepts_yes(self) -> None:
        session = mock.Mock(title="session one")
        with mock.patch("daydream.chat.sys.stdin.isatty", return_value=True), \
            mock.patch("daydream.chat.err_console.input", return_value="y"):
            self.assertTrue(_confirm_session_delete(session))

    def test_session_menu_delete_shortcut_returns_delete_action(self) -> None:
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

        sessions = [
            mock.Mock(title="First", model="qwen", updated_at=1.0, messages=[]),
            mock.Mock(title="Second", model="qwen", updated_at=2.0, messages=[]),
        ]
        with mock.patch("daydream.chat._raw_stdin", fake_raw), \
            mock.patch("daydream.chat._InlineTerminalRenderer", FakeRenderer), \
            mock.patch("daydream.chat.sys.stdin.isatty", return_value=True), \
            mock.patch("daydream.chat.err_console", mock.Mock(is_terminal=True)), \
            mock.patch("daydream.chat._read_key", side_effect=["d"]):
            action, chosen = _select_session_action(sessions, allow_delete=True)

        self.assertEqual(action, "delete")
        self.assertIs(chosen, sessions[0])

    def test_session_menu_ignores_delete_shortcut_when_not_allowed(self) -> None:
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

        sessions = [
            mock.Mock(title="First", model="qwen", updated_at=1.0, messages=[]),
            mock.Mock(title="Second", model="qwen", updated_at=2.0, messages=[]),
        ]
        with mock.patch("daydream.chat._raw_stdin", fake_raw), \
            mock.patch("daydream.chat._InlineTerminalRenderer", FakeRenderer), \
            mock.patch("daydream.chat.sys.stdin.isatty", return_value=True), \
            mock.patch("daydream.chat.err_console", mock.Mock(is_terminal=True)), \
            mock.patch("daydream.chat._read_key", side_effect=["d", "\r"]):
            action, chosen = _select_session_action(sessions, allow_delete=False)

        self.assertEqual(action, "resume")
        self.assertIs(chosen, sessions[0])

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

    def test_run_chat_applies_effort_after_new_session(self) -> None:
        captured_calls = []

        def fake_stream_response(_model, _tokenizer, request_messages, **kwargs):
            captured_calls.append((request_messages, kwargs))
            return ("hello", None, "")

        with mock.patch("daydream.chat._read_boxed_message", side_effect=["/new", "/effort long", "hello", "/quit"]), \
            mock.patch("daydream.chat.ensure_runtime_model", return_value="mlx-community/Qwen3-8B-4bit"), \
            mock.patch("daydream.chat.engine.load_model", return_value=("model", mock.Mock(has_thinking=True))), \
            mock.patch("daydream.chat._stream_response", side_effect=fake_stream_response), \
            mock.patch("daydream.chat.save_session"):
            run_chat("foo")

        self.assertEqual(len(captured_calls), 1)
        request_messages, kwargs = captured_calls[0]
        self.assertEqual(request_messages[0]["role"], "system")
        self.assertIn("Reasoning effort: long", request_messages[0]["content"])
        self.assertEqual(kwargs["chat_template_kwargs"], {"enable_thinking": True})

    def test_run_chat_applies_instant_effort_after_new_session(self) -> None:
        captured_calls = []

        def fake_stream_response(_model, _tokenizer, request_messages, **kwargs):
            captured_calls.append((request_messages, kwargs))
            return ("hello", None, "")

        with mock.patch("daydream.chat._read_boxed_message", side_effect=["/new", "/effort instant", "hello", "/quit"]), \
            mock.patch("daydream.chat.ensure_runtime_model", return_value="mlx-community/Qwen3-8B-4bit"), \
            mock.patch("daydream.chat.engine.load_model", return_value=("model", mock.Mock(has_thinking=True))), \
            mock.patch("daydream.chat._stream_response", side_effect=fake_stream_response), \
            mock.patch("daydream.chat.save_session"):
            run_chat("foo")

        self.assertEqual(len(captured_calls), 1)
        request_messages, kwargs = captured_calls[0]
        self.assertEqual(request_messages[0]["role"], "system")
        self.assertIn("Reasoning effort: instant", request_messages[0]["content"])
        self.assertEqual(kwargs["chat_template_kwargs"], {"enable_thinking": False})

    def test_run_chat_keeps_output_after_reasoning_in_memoryless_mode(self) -> None:
        captured_statuses: list[object] = []

        class FakeStatus:
            def __init__(self):
                self.output = ""
                self.had_reasoning = False
                self.reasoning_elapsed = None

            def start_reasoning(self):
                self.had_reasoning = True
                self.reasoning_elapsed = 1.0

            def append_reasoning(self, _text):
                return

            def end_reasoning(self):
                return

            def update(self, **_kwargs):
                return

            def ensure_minimum_wait(self, _seconds):
                return

            def append_output(self, text):
                self.output += text

        @contextmanager
        def fake_daydreaming_status(_console, _label):
            status = FakeStatus()
            captured_statuses.append(status)
            yield status

        @contextmanager
        def fake_status(_message):
            yield

        fake_err_console = mock.Mock(is_terminal=True)
        fake_err_console.status = fake_status
        fake_err_console.print = mock.Mock()
        fake_err_console.file = io.StringIO()
        fake_err_console.file.isatty = lambda: True

        responses = iter([
            SimpleNamespace(
                text="Here's a thinking process:\n1. Analyze.\n",
                prompt_tps=12.0,
                generation_tps=None,
                finish_reason=None,
            ),
            SimpleNamespace(
                text="</think>\nHello from Daydream.",
                prompt_tps=12.0,
                generation_tps=34.0,
                finish_reason="stop",
            ),
        ])

        with mock.patch("daydream.chat.err_console", fake_err_console), \
            mock.patch("daydream.chat._read_boxed_message", side_effect=["hello", "/quit"]), \
            mock.patch("daydream.chat.ensure_runtime_model", return_value="mlx-community/Foo-4bit"), \
            mock.patch("daydream.chat.engine.load_model", return_value=("model", mock.Mock(has_thinking=True))), \
            mock.patch("daydream.chat.engine.generate_stream", side_effect=lambda *args, **kwargs: responses), \
            mock.patch("daydream.chat.daydreaming_status", fake_daydreaming_status):
            run_chat("foo")

        self.assertEqual(len(captured_statuses), 1)
        self.assertIn("Hello from Daydream.", captured_statuses[0].output)
        self.assertNotIn("</think>", captured_statuses[0].output)
        raw = fake_err_console.file.getvalue()
        self.assertTrue(raw.startswith("\n" * _STATUS_OVERLAY_RESERVE_LINES))
        self.assertIn("\x1b[24;1H\n", raw)
        printed = " ".join(
            str(call.args[0])
            for call in fake_err_console.print.call_args_list
            if call.args
        )
        self.assertIn("Hello from Daydream.", printed)
        self.assertIn("Use /t to expand reasoning", printed)

    def test_run_chat_keeps_output_after_reasoning_in_persistent_session(self) -> None:
        captured_statuses: list[object] = []
        saved_sessions: list[object] = []

        class FakeStatus:
            def __init__(self):
                self.output = ""
                self.had_reasoning = False
                self.reasoning_elapsed = None

            def start_reasoning(self):
                self.had_reasoning = True
                self.reasoning_elapsed = 1.0

            def append_reasoning(self, _text):
                return

            def end_reasoning(self):
                return

            def update(self, **_kwargs):
                return

            def ensure_minimum_wait(self, _seconds):
                return

            def append_output(self, text):
                self.output += text

        @contextmanager
        def fake_daydreaming_status(_console, _label):
            status = FakeStatus()
            captured_statuses.append(status)
            yield status

        @contextmanager
        def fake_status(_message):
            yield

        def fake_save_session(session):
            saved_sessions.append(session)

        fake_err_console = mock.Mock(is_terminal=True)
        fake_err_console.status = fake_status
        fake_err_console.print = mock.Mock()
        fake_err_console.file = io.StringIO()
        fake_err_console.file.isatty = lambda: True

        responses = iter([
            SimpleNamespace(
                text="Here's a thinking process:\n1. Analyze.\n",
                prompt_tps=12.0,
                generation_tps=None,
                finish_reason=None,
            ),
            SimpleNamespace(
                text="</think>\nHello from persistent memory.",
                prompt_tps=12.0,
                generation_tps=34.0,
                finish_reason="stop",
            ),
        ])

        with mock.patch("daydream.chat.err_console", fake_err_console), \
            mock.patch("daydream.chat._read_boxed_message", side_effect=["/new", "hello", "/quit"]), \
            mock.patch("daydream.chat.ensure_runtime_model", return_value="mlx-community/Foo-4bit"), \
            mock.patch("daydream.chat.engine.load_model", return_value=("model", mock.Mock(has_thinking=True))), \
            mock.patch("daydream.chat.engine.generate_stream", side_effect=lambda *args, **kwargs: responses), \
            mock.patch("daydream.chat.daydreaming_status", fake_daydreaming_status), \
            mock.patch("daydream.chat.save_session", side_effect=fake_save_session):
            run_chat("foo")

        self.assertEqual(len(captured_statuses), 1)
        self.assertIn("Hello from persistent memory.", captured_statuses[0].output)
        self.assertNotIn("</think>", captured_statuses[0].output)
        raw = fake_err_console.file.getvalue()
        self.assertTrue(raw.startswith("\n" * _STATUS_OVERLAY_RESERVE_LINES))
        self.assertIn("\x1b[24;1H\n", raw)
        printed = " ".join(
            str(call.args[0])
            for call in fake_err_console.print.call_args_list
            if call.args
        )
        self.assertIn("Hello from persistent memory.", printed)
        self.assertIn("Use /t to expand reasoning", printed)
        self.assertTrue(saved_sessions)
        self.assertEqual(saved_sessions[-1].messages[-1].content, "Hello from persistent memory.")

    def test_run_chat_resume_prints_saved_history(self) -> None:
        from daydream.storage import ChatMessage, ChatSession

        fake_err_console = mock.Mock(is_terminal=True)
        fake_err_console.print = mock.Mock()
        fake_err_console.status = contextmanager(lambda _msg: iter([None]))  # type: ignore[arg-type]

        session = ChatSession(
            session_id="abc123",
            model="qwen3.5-9b",
            title="Saved chat",
            created_at=1.0,
            updated_at=2.0,
            messages=[
                ChatMessage(role="user", content="Earlier question", timestamp=1.0),
                ChatMessage(role="assistant", content="Earlier answer", timestamp=2.0, reasoning="hidden chain"),
            ],
            memories=[],
        )

        with mock.patch("daydream.chat.err_console", fake_err_console), \
            mock.patch("daydream.chat._read_boxed_message", side_effect=["/resume", "/quit"]), \
            mock.patch("daydream.chat.ensure_runtime_model", return_value="mlx-community/Qwen3.5-9B-4bit"), \
            mock.patch("daydream.chat.engine.load_model", return_value=("model", mock.Mock(has_thinking=True))), \
            mock.patch("daydream.chat.list_sessions", return_value=[session]), \
            mock.patch("daydream.chat.load_memories", return_value=[]), \
            mock.patch("daydream.chat._select_session_action", return_value=("resume", session)):
            run_chat("foo", display_name="qwen3.5-9b")

        printed = " ".join(
            str(call.args[0])
            for call in fake_err_console.print.call_args_list
            if call.args
        )
        self.assertIn("Resumed: Saved chat", printed)
        self.assertIn("Earlier question", printed)
        self.assertIn("Earlier answer", printed)
        self.assertNotIn("hidden chain", printed)


if __name__ == "__main__":
    unittest.main()

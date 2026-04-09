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
    InlineFlowRenderer,
    _InlineTerminalRenderer,
    _ReasoningParser,
    _build_request_messages,
    _build_session_memory_prompt,
    _clean_final_output_from_reasoning_leak,
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
    _iter_body_chunks,
    _merge_escape_key,
    _matching_slash_commands,
    _make_menu_renderer,
    _model_supports_effort,
    _normalize_cli_page_mode,
    _normalize_effort,
    _read_key,
    _read_live_boxed_message,
    _extract_answer_labeled_text,
    _recover_final_output,
    _repack_tight_page,
    _select_cli_page_mode,
    _select_dreaming_mode,
    _select_effort,
    _select_session_action,
    _status_overlay_reserve_lines,
    _transcript_blocks_from_messages,
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

    def test_reasoning_parser_switches_to_reasoning_after_visible_output_on_strong_marker(self) -> None:
        parser = _ReasoningParser()

        delta, reasoning_delta, closed = parser.feed("Hello there.\n\n")
        self.assertEqual(delta, "Hello there.\n\n")
        self.assertEqual(reasoning_delta, "")
        self.assertFalse(closed)

        delta, reasoning_delta, closed = parser.feed("*Refinement:* Keep it conversational.\n")
        self.assertEqual(delta, "")
        self.assertIn("*Refinement:* Keep it conversational.", reasoning_delta)
        self.assertFalse(closed)

    def test_reasoning_parser_keeps_thinking_block_response_inside_reasoning(self) -> None:
        parser = _ReasoningParser()

        delta, reasoning_delta, closed = parser.feed(
            "Thinking block:\n"
            "1. User repeats hi.\n"
            "Response:\n"
            "Hello again! Ready to help?\n\n"
            "Hello again! What are we working on?"
        )
        self.assertEqual(delta, "Hello again! What are we working on?")
        self.assertIn("Thinking block:", reasoning_delta)
        self.assertIn("Response:\nHello again! Ready to help?", reasoning_delta)
        self.assertTrue(closed)

    def test_reasoning_parser_hides_meta_thinking_trace_before_final_answer(self) -> None:
        parser = _ReasoningParser()

        delta, reasoning_delta, closed = parser.feed(
            "Okay, I will count the lines in the thinking block.\n\n"
            "That's 4 lines. Perfect.\n\n"
            "Now the actual response.\n"
            "\"Hello! I'm Qwen, a large language model. How can I assist you today?\"\n\n"
            "1. Identify user intent: Self-introduction.\n"
            "2. Recall identity: Qwen, large language model.\n"
            "3. Keep response brief and friendly.\n"
            "4. Ignore typo in user query for clarity.\n\n"
            "Hello! I'm Qwen, a large language model trained by Tongyi Lab. How can I assist you today?"
        )
        self.assertEqual(
            delta,
            "Hello! I'm Qwen, a large language model trained by Tongyi Lab. How can I assist you today?",
        )
        self.assertIn("thinking block", reasoning_delta.lower())
        self.assertIn("Now the actual response.", reasoning_delta)
        self.assertTrue(closed)

    def test_reasoning_parser_hides_usually_means_and_instruction_check_trace(self) -> None:
        parser = _ReasoningParser()

        delta, reasoning_delta, closed = parser.feed(
            "Usually, this means I should output the reasoning.\n"
            "Let's check the system instruction again: \"Keep your thinking block under 3-5 lines.\"\n"
            "Hello! How can I help you today?"
        )
        self.assertEqual(delta, "Hello! How can I help you today?")
        self.assertIn("Usually, this means", reasoning_delta)
        self.assertIn("Let's check the system instruction again", reasoning_delta)
        self.assertTrue(closed)

    def test_matching_slash_commands_filters_by_prefix(self) -> None:
        matches = _matching_slash_commands("/e")
        self.assertEqual(matches[0][0], "/effort")
        self.assertTrue(any(name == "/help" for name, _ in _matching_slash_commands("/")))
        self.assertTrue(any(name == "/cli-page" for name, _ in _matching_slash_commands("/cli")))

    def test_normalize_cli_page_mode_accepts_supported_values(self) -> None:
        self.assertEqual(_normalize_cli_page_mode("TIGHT"), "tight")
        self.assertEqual(_normalize_cli_page_mode(" loose "), "loose")
        self.assertIsNone(_normalize_cli_page_mode("compact"))

    def test_current_command_selection_defaults_and_preserves(self) -> None:
        matches, selected = _current_command_selection("/", None, multiline=False)
        self.assertEqual(selected, "/effort")
        self.assertTrue(matches)

        matches, selected = _current_command_selection("/e", "/effort", multiline=False)
        self.assertEqual(selected, "/effort")
        self.assertEqual(matches[0][0], "/effort")

    def test_make_menu_renderer_uses_bottom_renderer_in_tight_mode(self) -> None:
        renderer = _make_menu_renderer("tight")
        self.assertIsInstance(renderer, InlineFlowRenderer)

    def test_iter_body_chunks_splits_confirmed_body_into_small_pieces(self) -> None:
        chunks = _iter_body_chunks("Hello from Daydream.")
        self.assertGreater(len(chunks), 5)
        self.assertEqual("".join(chunks), "Hello from Daydream.")
        self.assertEqual(chunks[:5], list("Hello"))

    def test_iter_body_chunks_resets_to_single_chars_after_newline(self) -> None:
        chunks = _iter_body_chunks("Hello there.\nWorld again.")
        newline_index = chunks.index("\n")
        self.assertEqual(chunks[newline_index + 1:newline_index + 6], list("World"))

    def test_transcript_blocks_from_messages_adds_reasoning_summary_for_assistant(self) -> None:
        from daydream.storage import ChatMessage

        blocks = _transcript_blocks_from_messages([
            ChatMessage(role="user", content="hi", timestamp=1.0),
            ChatMessage(role="assistant", content="hello", timestamp=2.0, reasoning="hidden"),
        ])
        self.assertEqual([block.kind for block in blocks], ["user", "reasoning_summary", "assistant"])

    def test_repack_tight_page_reprints_compact_history(self) -> None:
        fake_console = mock.Mock(is_terminal=True)
        fake_console.print = mock.Mock()
        fake_stream = io.StringIO()
        fake_stream.isatty = lambda: True
        fake_console.file = fake_stream

        with mock.patch("daydream.chat.err_console", fake_console):
            _repack_tight_page(
                "qwen3.5-9b",
                [
                    mock.Mock(kind="user", content="hi", elapsed=None),
                    mock.Mock(kind="reasoning_summary", content="", elapsed=1.5),
                    mock.Mock(kind="assistant", content="hello", elapsed=None),
                ],
            )

        self.assertTrue(fake_stream.getvalue().startswith("\x1b[2J\x1b[H"))
        printed = " ".join(str(call.args[0]) for call in fake_console.print.call_args_list if call.args)
        self.assertIn("Use / for commands.", printed)
        self.assertIn("hi", printed)
        self.assertIn("Daydreamed for 1.5s", printed)
        self.assertIn("hello", printed)

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

        self.assertEqual(result, "/cli-page")

    def test_live_boxed_message_does_not_promote_terminal_after_user_input(self) -> None:
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

        fake_err_console = mock.Mock()
        fake_err_console.print = mock.Mock()

        with mock.patch("daydream.chat._raw_stdin", fake_raw), \
            mock.patch("daydream.chat._InlineTerminalRenderer", FakeRenderer), \
            mock.patch("daydream.chat._read_key", side_effect=["h", "i", "\r"]), \
            mock.patch("daydream.chat.err_console", fake_err_console), \
            mock.patch("daydream.chat._promote_terminal_output_to_scrollback") as promote:
            result = _read_live_boxed_message()

        self.assertEqual(result, "hi")
        promote.assert_not_called()

    def test_clean_final_output_keeps_only_last_response_after_reasoning_blocks(self) -> None:
        text = (
            "This implies I should output the reasoning *and* the answer, but keep the reasoning very short.\n\n"
            "*Revised Plan:*\n"
            "1. Identify I am an AI trained by Google.\n"
            "2. State that clearly.\n\n"
            "*Thinking Block:*\n"
            "I am an AI assistant trained by Google.\n"
            "My purpose is to assist with tasks and answer questions.\n"
            "I do not have personal identity or feelings.\n\n"
            "*Answer:*\n"
            "I am a large language model, trained by Google.\n\n"
            "*Constraint Check:*\n"
            "Thinking block lines: 3.\n\n"
            "I am an AI assistant trained by Google.\n"
            "I process text to generate responses.\n"
            "I do not have a physical form.\n\n"
            "I am a large language model, trained by Google."
        )
        self.assertEqual(
            _clean_final_output_from_reasoning_leak(text),
            "I am a large language model, trained by Google.",
        )

    def test_clean_final_output_keeps_only_last_response_for_meta_greeting_trace(self) -> None:
        text = (
            "User says hello.\n"
            "Intent is greeting/connection.\n"
            "Response should be friendly + offer help.\n"
            "Keep it short.\n\n"
            "*Draft:*\n"
            "Hello! How can I assist you today?\n"
            "That's a simple greeting, so I'll just reply warmly.\n"
            "Ready for your next question.\n\n"
            "*Final Answer:* Hello! How can I help you today?\n\n"
            "Actually, looking at the instruction again: \"Keep your thinking block under 3-5 lines.\" This is a directive for the output if the thinking block is rendered. I will ensure the text I generate as \"thought\" is short.\n\n"
            "Let's just respond naturally but concisely.\n\n"
            "*Thinking:*\n"
            "1. User initiates with \"hello\".\n"
            "2. Standard protocol is reciprocal greeting + offer assistance.\n"
            "3. No complex reasoning required for a simple greeting.\n"
            "4. Drafting a brief, friendly reply.\n\n"
            "*Response:* Hello! How can I help you today?\n\n"
            "*Revised Plan:* Just answer.\n"
            "User greeted with \"hello\".\n"
            "Appropriate response is a greeting plus offer of help.\n"
            "Keep response brief to adhere to conciseness.\n\n"
            "*Response:* Hello! How can I help you today?\n\n"
            "Hello! How can I help you today?"
        )
        self.assertEqual(
            _clean_final_output_from_reasoning_leak(text),
            "Hello! How can I help you today?",
        )

    def test_clean_final_output_keeps_only_last_response_for_meta_qwen_trace(self) -> None:
        text = (
            "Okay, I will count the lines in the thinking block.\n\n"
            "That's 4 lines. Perfect.\n\n"
            "Now the actual response.\n"
            "\"Hello! I'm Qwen, a large language model. How can I assist you today?\"\n\n"
            "1. Identify user intent: Self-introduction.\n"
            "2. Recall identity: Qwen, large language model.\n"
            "3. Keep response brief and friendly.\n"
            "4. Ignore typo in user query for clarity.\n\n"
            "Hello! I'm Qwen, a large language model trained by Tongyi Lab. How can I assist you today?"
        )
        self.assertEqual(
            _clean_final_output_from_reasoning_leak(text),
            "Hello! I'm Qwen, a large language model trained by Tongyi Lab. How can I assist you today?",
        )

    def test_clean_final_output_keeps_only_last_response_after_instruction_check_trace(self) -> None:
        text = (
            "Usually, this means I should output the reasoning.\n"
            "Let's check the system instruction again: \"Keep your thinking block under 3-5 lines.\"\n\n"
            "Hello! I'm Qwen, a large language model. How can I assist you today?\n\n"
            "1. Identify user intent: Self-introduction.\n"
            "2. Recall identity: Qwen, large language model.\n"
            "3. Keep response brief and friendly.\n"
            "4. Ignore typo in user query for clarity.\n\n"
            "Hello! I'm Qwen, a large language model trained by Tongyi Lab. How can I assist you today?"
        )
        self.assertEqual(
            _clean_final_output_from_reasoning_leak(text),
            "Hello! I'm Qwen, a large language model trained by Tongyi Lab. How can I assist you today?",
        )

    def test_recover_final_output_falls_back_to_last_non_reasoning_paragraph(self) -> None:
        reasoning = (
            "Usually, this means I should output the reasoning.\n\n"
            "Let's check the system instruction again: \"Keep your thinking block under 3-5 lines.\"\n\n"
            "Hello! I'm Qwen, a large language model trained by Tongyi Lab. How can I assist you today?"
        )
        self.assertEqual(
            _recover_final_output("", reasoning),
            "Hello! I'm Qwen, a large language model trained by Tongyi Lab. How can I assist you today?",
        )

    def test_extract_answer_labeled_text_supports_chinese_answer_labels(self) -> None:
        text = "分析：用户发送“？”意在确认响应状态。\n回答：我在，有什么可以帮您的吗？"
        self.assertEqual(
            _extract_answer_labeled_text(text),
            "我在，有什么可以帮您的吗？",
        )

    def test_clean_final_output_keeps_only_chinese_answer_after_analysis_block(self) -> None:
        text = "分析：用户发送“？”意在确认响应状态。\n回答：我在，有什么可以帮您的吗？"
        self.assertEqual(
            _clean_final_output_from_reasoning_leak(text),
            "我在，有什么可以帮您的吗？",
        )

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

    def test_effort_menu_escape_cancels_immediately(self) -> None:
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
            mock.patch("daydream.chat._read_key", return_value="\x1b"):
            result = _select_effort("default", supported=True)

        self.assertEqual(result, "default")

    def test_cli_page_menu_selects_tight_mode(self) -> None:
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
            mock.patch("daydream.chat._read_key", side_effect=["\x1b", "[", "B", "\r"]):
            result = _select_cli_page_mode("loose")

        self.assertEqual(result, "tight")

    def test_cli_page_menu_escape_cancels_immediately(self) -> None:
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
            mock.patch("daydream.chat._read_key", return_value="\x1b"):
            result = _select_cli_page_mode("loose")

        self.assertEqual(result, "loose")

    def test_dreaming_menu_escape_cancels_immediately(self) -> None:
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
            mock.patch("daydream.chat._read_key", return_value="\x1b"):
            result = _select_dreaming_mode()

        self.assertIsNone(result)

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
        self.assertIn(
            "Reason briefly and efficiently",
            _effort_system_prompt("short", "mlx-community/Qwen3-8B-4bit"),
        )
        self.assertIsNone(_effort_system_prompt("short", "mlx-community/SmolLM2-360M-Instruct-4bit"))
        tokenizer = mock.Mock(has_thinking=True)
        self.assertEqual(_effort_chat_template_kwargs("instant", tokenizer), {"enable_thinking": False})
        self.assertEqual(_effort_chat_template_kwargs("short", tokenizer), {"enable_thinking": True, "thinking_budget": 200})
        self.assertEqual(_effort_chat_template_kwargs("long", tokenizer), {"enable_thinking": True, "thinking_budget": 10000})
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
        self.assertIn("Reason thoroughly before answering", request[0]["content"])
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
        self.assertIn("Answer directly and concisely", request[0]["content"])

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

    def test_session_menu_escape_cancels_immediately(self) -> None:
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
            mock.patch("daydream.chat._read_key", return_value="\x1b"):
            result = _select_session_action(sessions, allow_delete=False)

        self.assertIsNone(result)

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

    def test_inline_terminal_renderer_reflows_on_first_render_and_shrink(self) -> None:
        stream = io.StringIO()

        class Renderer(_InlineTerminalRenderer):
            def __init__(self, stream):
                super().__init__(stream)
                self.rows = 24
                self.cols = 96

            def _terminal_size(self):
                return os.terminal_size((self.cols, self.rows))

        renderer = Renderer(stream)
        renderer.render(["a", "b", "c"])
        first_snapshot = stream.getvalue()
        renderer.render(["a", "b", "c", "d", "e", "f"])
        growth_snapshot = stream.getvalue()
        renderer.render(["a", "b", "c"])
        renderer.finish()

        growth_output = growth_snapshot[len(first_snapshot):]
        shrink_output = stream.getvalue()[len(growth_snapshot):]

        self.assertIn("\x1b[24;1H\n\n\n", first_snapshot)
        self.assertIn("\x1b[22;1H\x1b[J", growth_output)
        self.assertIn("\x1b[24;1H\n\n\n", growth_output)
        self.assertIn("\x1b[19;1H\x1b[J", shrink_output)
        self.assertIn("\x1b[1;1H\x1b[3L", shrink_output)

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
        self.assertIn("Reason thoroughly before answering", request_messages[0]["content"])
        self.assertEqual(kwargs["chat_template_kwargs"], {"enable_thinking": True, "thinking_budget": 10000})

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
        self.assertIn("Answer directly and concisely", request_messages[0]["content"])
        self.assertEqual(kwargs["chat_template_kwargs"], {"enable_thinking": False})

    def test_run_chat_cli_page_updates_default_mode(self) -> None:
        with mock.patch("daydream.chat._read_boxed_message", side_effect=["/cli-page tight", "/quit"]), \
            mock.patch("daydream.chat.ensure_runtime_model", return_value="mlx-community/Qwen3-8B-4bit"), \
            mock.patch("daydream.chat.engine.load_model", return_value=("model", mock.Mock(has_thinking=True))), \
            mock.patch("daydream.chat.get_default_cli_page_mode", return_value="loose"), \
            mock.patch("daydream.chat.set_default_cli_page_mode", return_value="tight") as set_mode, \
            mock.patch("daydream.chat.err_console") as fake_console:
            fake_console.is_terminal = True
            run_chat("foo")

        set_mode.assert_called_once_with("tight")
        printed = " ".join(
            str(call.args[0])
            for call in fake_console.print.call_args_list
            if call.args
        )
        self.assertIn("CLI page mode set to tight", printed)

    def test_run_chat_repacks_page_after_response_in_tight_mode(self) -> None:
        with mock.patch("daydream.chat._read_boxed_message", side_effect=["hello", "/quit"]), \
            mock.patch("daydream.chat.ensure_runtime_model", return_value="mlx-community/Qwen3-8B-4bit"), \
            mock.patch("daydream.chat.engine.load_model", return_value=("model", mock.Mock(has_thinking=True))), \
            mock.patch("daydream.chat.get_default_cli_page_mode", return_value="tight"), \
            mock.patch("daydream.chat._stream_response", return_value=("Hi there.", None, "reasoning", 2.0)), \
            mock.patch("daydream.chat._repack_tight_page") as repack:
            run_chat("foo", display_name="qwen3.5-9b")

        repack.assert_called_once()
        args = repack.call_args.args
        self.assertEqual(args[0], "qwen3.5-9b")
        kinds = [block.kind for block in args[1]]
        self.assertEqual(kinds, ["user", "reasoning_summary", "assistant", "reasoning_hint"])

    def test_run_chat_repacks_resumed_history_in_tight_mode(self) -> None:
        from daydream.storage import ChatMessage, ChatSession

        session = ChatSession(
            session_id="abc123",
            model="qwen3.5-9b",
            title="Saved chat",
            created_at=1.0,
            updated_at=2.0,
            messages=[
                ChatMessage(role="user", content="Earlier question", timestamp=1.0),
                ChatMessage(role="assistant", content="Earlier answer", timestamp=2.0, reasoning="hidden"),
            ],
            memories=[],
        )

        with mock.patch("daydream.chat._read_boxed_message", side_effect=["/resume", "/quit"]), \
            mock.patch("daydream.chat.ensure_runtime_model", return_value="mlx-community/Qwen3.5-9B-4bit"), \
            mock.patch("daydream.chat.engine.load_model", return_value=("model", mock.Mock(has_thinking=True))), \
            mock.patch("daydream.chat.get_default_cli_page_mode", return_value="tight"), \
            mock.patch("daydream.chat.list_sessions", return_value=[session]), \
            mock.patch("daydream.chat.load_memories", return_value=[]), \
            mock.patch("daydream.chat._select_session_action", return_value=("resume", session)), \
            mock.patch("daydream.chat._repack_tight_page") as repack:
            run_chat("foo", display_name="qwen3.5-9b")

        repack.assert_called_once()
        blocks = repack.call_args.args[1]
        self.assertEqual([block.kind for block in blocks], ["user", "reasoning_summary", "assistant"])

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
        def fake_daydreaming_status(_console, _label, **_kwargs):
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
        self.assertEqual(captured_statuses[0].output, "")
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
        def fake_daydreaming_status(_console, _label, **_kwargs):
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
        self.assertEqual(captured_statuses[0].output, "")
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

    def test_run_chat_resume_prints_only_last_ten_visible_messages(self) -> None:
        from daydream.storage import ChatMessage, ChatSession

        fake_err_console = mock.Mock(is_terminal=True)
        fake_err_console.print = mock.Mock()
        fake_err_console.status = contextmanager(lambda _msg: iter([None]))  # type: ignore[arg-type]

        saved_messages = [
            ChatMessage(role="user", content=f"Question-{index:02d}", timestamp=float(index))
            if index % 2 == 0
            else ChatMessage(role="assistant", content=f"Answer-{index:02d}", timestamp=float(index))
            for index in range(12)
        ]
        session = ChatSession(
            session_id="many123",
            model="qwen3.5-9b",
            title="Long chat",
            created_at=1.0,
            updated_at=2.0,
            messages=saved_messages,
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
        self.assertNotIn("Question-00", printed)
        self.assertNotIn("Answer-01", printed)
        self.assertIn("Question-02", printed)
        self.assertIn("Answer-11", printed)

    def test_run_chat_strips_reasoning_leak_from_final_output(self) -> None:
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
        def fake_daydreaming_status(_console, _label, **_kwargs):
            status = FakeStatus()
            captured_statuses.append(status)
            yield status

        fake_err_console = mock.Mock(is_terminal=True)
        fake_err_console.print = mock.Mock()
        fake_err_console.status = contextmanager(lambda _msg: iter([None]))  # type: ignore[arg-type]
        fake_err_console.file = io.StringIO()
        fake_err_console.file.isatty = lambda: True

        responses = iter([
            SimpleNamespace(
                text="你好！很高兴为您服务。\n我可以用中文和您交流。有什么我可以帮您的吗？\n\n",
                prompt_tps=12.0,
                generation_tps=18.0,
                finish_reason=None,
            ),
            SimpleNamespace(
                text="*Refinement:* Keep it conversational.\n\"你好！很高兴见到你。有什么我可以帮助你的吗？\"\n\n",
                prompt_tps=12.0,
                generation_tps=18.0,
                finish_reason=None,
            ),
            SimpleNamespace(
                text="*Final Output:*\n你好！很高兴和你打招呼。有什么我可以帮你的吗？",
                prompt_tps=12.0,
                generation_tps=18.0,
                finish_reason="stop",
            ),
        ])

        with mock.patch("daydream.chat.err_console", fake_err_console), \
            mock.patch("daydream.chat._read_boxed_message", side_effect=["你好", "/quit"]), \
            mock.patch("daydream.chat.ensure_runtime_model", return_value="mlx-community/Foo-4bit"), \
            mock.patch("daydream.chat.engine.load_model", return_value=("model", mock.Mock(has_thinking=True))), \
            mock.patch("daydream.chat.engine.generate_stream", side_effect=lambda *args, **kwargs: responses), \
            mock.patch("daydream.chat.daydreaming_status", fake_daydreaming_status):
            run_chat("foo")

        printed = " ".join(
            str(call.args[0])
            for call in fake_err_console.print.call_args_list
            if call.args
        )
        self.assertIn("你好！很高兴和你打招呼。有什么我可以帮你的吗？", printed)
        self.assertNotIn("Refinement", printed)
        self.assertNotIn("Final Output", printed)
        self.assertNotIn("我可以用中文和您交流", printed)

    def test_run_chat_does_not_duplicate_final_output_after_reasoning(self) -> None:
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

            def clear_output(self):
                self.output = ""

        @contextmanager
        def fake_daydreaming_status(_console, _label, **_kwargs):
            yield FakeStatus()

        fake_err_console = mock.Mock(is_terminal=True)
        fake_err_console.print = mock.Mock()
        fake_err_console.status = contextmanager(lambda _msg: iter([None]))  # type: ignore[arg-type]
        fake_err_console.file = io.StringIO()
        fake_err_console.file.isatty = lambda: True

        responses = iter([
            SimpleNamespace(
                text="<think>analysis</think>\n好的，随时待命！接下来你想聊些什么，或者有什么具体的问题需要我协助吗？",
                prompt_tps=12.0,
                generation_tps=18.0,
                finish_reason="stop",
            ),
        ])

        with mock.patch("daydream.chat.err_console", fake_err_console), \
            mock.patch("daydream.chat._read_boxed_message", side_effect=["你好", "/quit"]), \
            mock.patch("daydream.chat.ensure_runtime_model", return_value="mlx-community/Foo-4bit"), \
            mock.patch("daydream.chat.engine.load_model", return_value=("model", mock.Mock(has_thinking=True))), \
            mock.patch("daydream.chat.engine.generate_stream", side_effect=lambda *args, **kwargs: responses), \
            mock.patch("daydream.chat.daydreaming_status", fake_daydreaming_status):
            run_chat("foo")

        printed = " ".join(
            str(call.args[0])
            for call in fake_err_console.print.call_args_list
            if call.args
        )
        self.assertEqual(
            printed.count("好的，随时待命！接下来你想聊些什么，或者有什么具体的问题需要我协助吗？"),
            1,
        )

    def test_run_chat_streams_visible_text_after_reasoning_to_console(self) -> None:
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

            def clear_output(self):
                self.output = ""

            def stop(self):
                return

        @contextmanager
        def fake_daydreaming_status(_console, _label, **_kwargs):
            status = FakeStatus()
            captured_statuses.append(status)
            yield status

        fake_err_console = mock.Mock(is_terminal=True)
        fake_err_console.print = mock.Mock()
        fake_err_console.status = contextmanager(lambda _msg: iter([None]))  # type: ignore[arg-type]
        fake_err_console.file = io.StringIO()
        fake_err_console.file.isatty = lambda: True

        responses = iter([
            SimpleNamespace(
                text="<think>analysis</think>\nFirst line.",
                prompt_tps=12.0,
                generation_tps=18.0,
                finish_reason=None,
            ),
            SimpleNamespace(
                text=" Second line.",
                prompt_tps=12.0,
                generation_tps=18.0,
                finish_reason="stop",
            ),
        ])

        with mock.patch("daydream.chat.err_console", fake_err_console), \
            mock.patch("daydream.chat._read_boxed_message", side_effect=["你好", "/quit"]), \
            mock.patch("daydream.chat.ensure_runtime_model", return_value="mlx-community/Foo-4bit"), \
            mock.patch("daydream.chat.engine.load_model", return_value=("model", mock.Mock(has_thinking=True))), \
            mock.patch("daydream.chat.engine.generate_stream", side_effect=lambda *args, **kwargs: responses), \
            mock.patch("daydream.chat.daydreaming_status", fake_daydreaming_status):
            run_chat("foo")

        self.assertEqual(len(captured_statuses), 1)
        self.assertEqual(captured_statuses[0].output, "")
        printed = " ".join(
            str(call.args[0])
            for call in fake_err_console.print.call_args_list
            if call.args
        )
        self.assertIn("First line. Second line.", printed)


if __name__ == "__main__":
    unittest.main()

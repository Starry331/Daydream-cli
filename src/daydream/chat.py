"""Interactive chat REPL and one-shot generation."""

from __future__ import annotations

import os
import select
import shutil
import sys
import re
import termios
import time
import tty
from contextlib import contextmanager
from typing import Callable

from rich.console import Console

from daydream import engine
from daydream.models import ensure_runtime_model
from daydream.registry import reverse_lookup
from daydream.utils import (
    BottomTerminalRenderer,
    build_dreaming_menu_lines,
    build_effort_menu_lines,
    build_input_box_lines,
    build_memory_display_lines,
    build_session_list_lines,
    daydreaming_status,
    DreamingStatus,
    render_expanded_reasoning,
    render_input_box,
)
from daydream.storage import (
    ChatSession,
    ChatMessage,
    Memory,
    save_session,
    load_session,
    list_sessions,
    save_memories,
    load_memories,
)
from daydream.dreaming import run_reming, run_dreaming

console = Console()
err_console = Console(stderr=True)

MULTILINE_SENTINEL = '"""'
OPEN_THINK_TAG = "<think>"
CLOSE_THINK_TAG = "</think>"
EFFORT_LEVELS = ("instant", "short", "default", "long")
SLASH_COMMANDS = (
    ("/effort", "adjust reasoning depth"),
    ("/help", "show available commands"),
    ("/new", "start a persistent chat session"),
    ("/resume", "resume a saved session"),
    ("/dreaming", "consolidate memories from conversation"),
    ("/memory", "view extracted memories"),
    ("/reset", "clear conversation history"),
    ("/clear", "alias for /reset"),
    ("/t", "expand the last reasoning trace"),
    ("/quit", "exit chat"),
)
REASONING_PREFIX_CANDIDATES = (
    "thinking process",
    "thinking process:",
    "here's a thinking process",
    "here is a thinking process",
    "the user asked",
    "the user said",
    "the user sent",
    "i need to",
    "i should",
    "first,",
    "however,",
    "in summary",
    "reasoning",
    "reasoning:",
    "analysis:",
    "let's think",
    "思考过程",
    "推理过程",
    "让我想",
    "好的，用户",
    "好的，用户发来",
    "好的，用户发来的是",
    "用户发来",
    "用户发来的是",
    "用户可能",
    "用户希望",
    "看起来他们可能",
    "我需要",
    "我应该",
    "首先",
    "不过",
    "另外",
    "同时",
    "现在需要",
    "根据我的知识库",
    "总结",
    "总结：",
)


def _print_stats(response):
    """Print generation statistics."""
    parts = []
    if response.prompt_tokens:
        parts.append(f"prompt: {response.prompt_tokens} tokens")
    if response.prompt_tps:
        parts.append(f"prompt: {response.prompt_tps:.1f} t/s")
    if response.generation_tokens:
        parts.append(f"gen: {response.generation_tokens} tokens")
    if response.generation_tps:
        parts.append(f"gen: {response.generation_tps:.1f} t/s")
    if response.peak_memory:
        parts.append(f"mem: {response.peak_memory:.1f} GB")
    if parts:
        err_console.print(f"[dim]  {' | '.join(parts)}[/dim]")


def _collect_multiline_message(
    first_line: str,
    input_func: Callable[[str], str] = input,
) -> str:
    """Collect a multi-line user message from the REPL."""
    stripped = first_line.strip()

    if stripped == MULTILINE_SENTINEL:
        lines: list[str] = []
        while True:
            line = input_func("... ")
            if line.strip() == MULTILINE_SENTINEL:
                break
            lines.append(line)
        return "\n".join(lines).strip()

    if first_line.endswith("\\"):
        lines = [first_line[:-1].rstrip()]
        while True:
            line = input_func("... ")
            if line.endswith("\\"):
                lines.append(line[:-1].rstrip())
                continue
            lines.append(line)
            break
        return "\n".join(lines).strip()

    return first_line


def _read_boxed_message(
    *,
    input_func: Callable[[str], str] = input,
) -> str:
    if input_func is input and sys.stdin.isatty() and err_console.is_terminal:
        return _read_live_boxed_message()

    err_console.print(render_input_box([""], placeholder="Type a message or / for commands"))
    try:
        first_line = input_func("│ ")
        return _collect_multiline_message(first_line, lambda _: input_func("│ "))
    finally:
        err_console.print()


def _matching_slash_commands(text: str) -> list[tuple[str, str]]:
    sample = text.strip()
    if not sample.startswith("/"):
        return []
    if sample == "/":
        return list(SLASH_COMMANDS)

    command_prefix = sample.split(None, 1)[0].lower()
    matches = [row for row in SLASH_COMMANDS if row[0].startswith(command_prefix)]
    return matches or list(SLASH_COMMANDS)


def _normalize_effort(value: str) -> str | None:
    choice = value.strip().lower()
    if choice in EFFORT_LEVELS:
        return choice
    return None


def _model_supports_effort(model_name: str) -> bool:
    lowered = model_name.lower()
    return any(
        marker in lowered
        for marker in (
            "qwen3",
            "deepseek-r1",
            "reasoner",
            "thinking",
        )
    )


def _effort_chat_template_kwargs(effort: str, tokenizer) -> dict:
    if effort == "default":
        return {}
    has_thinking = bool(getattr(tokenizer, "has_thinking", False))
    if not has_thinking:
        return {}
    if effort == "instant":
        return {"enable_thinking": False}
    return {"enable_thinking": True}


def _effort_system_prompt(effort: str, model_name: str) -> str | None:
    if effort == "default" or not _model_supports_effort(model_name):
        return None

    prompts = {
        "instant": (
            "Reasoning effort: instant. Answer directly. Do not emit a thinking block, "
            "do not spend extra tokens on hidden reasoning, and get to the answer fast."
        ),
        "short": (
            "Reasoning effort: short. If you reason, keep the reasoning compact and minimal "
            "before answering."
        ),
        "long": (
            "Reasoning effort: long. Spend noticeably more time reasoning, explore edge cases, "
            "and use a fuller thinking block before producing the final answer."
        ),
    }
    return prompts.get(effort)


def _build_request_messages(
    history: list[dict],
    *,
    system_prompt: str | None,
    effort: str,
    model_name: str,
    session_memories: list[Memory] | None = None,
) -> list[dict]:
    request_messages: list[dict] = []
    if system_prompt:
        request_messages.append({"role": "system", "content": system_prompt})
    memory_prompt = _build_session_memory_prompt(session_memories or [])
    if memory_prompt:
        request_messages.append({"role": "system", "content": memory_prompt})
    effort_prompt = _effort_system_prompt(effort, model_name)
    if effort_prompt:
        request_messages.append({"role": "system", "content": effort_prompt})
    request_messages.extend(history)
    return request_messages


def _build_session_memory_prompt(memories: list[Memory]) -> str | None:
    if not memories:
        return None

    unique: dict[str, Memory] = {}
    ordered = sorted(
        memories,
        key=lambda memory: (-memory.importance, memory.created_at, memory.content),
    )
    for memory in ordered:
        key = memory.content.strip().lower()
        if key and key not in unique:
            unique[key] = memory

    lines = [
        "Session memory for this persistent chat only.",
        "Use these memories when relevant, but do not mention this memory block unless it helps answer the user.",
    ]
    for memory in list(unique.values())[:12]:
        content = memory.content.strip().replace("\n", " ")
        if len(content) > 180:
            content = content[:177] + "..."
        lines.append(f"- [{memory.category} | {memory.importance:.2f}] {content}")

    return "\n".join(lines)


def _confirm_memory_import(memories: list[Memory]) -> bool:
    if not memories:
        return False

    count = len(memories)
    noun = "memory" if count == 1 else "memories"
    while True:
        reply = err_console.input(
            f"[dim]Add {count} extracted {noun} to this session memory only? [Y/n][/dim] "
        ).strip().lower()
        if reply in ("", "y", "yes"):
            return True
        if reply in ("n", "no"):
            return False
        err_console.print("[dim]Please answer Y or n.[/dim]")


@contextmanager
def _raw_stdin():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _read_key() -> str:
    fd = sys.stdin.fileno()
    first_byte = os.read(fd, 1)
    if not first_byte:
        return ""

    if first_byte != b"\x1b":
        expected_length = 1
        value = first_byte[0]
        if value & 0b11110000 == 0b11110000:
            expected_length = 4
        elif value & 0b11100000 == 0b11100000:
            expected_length = 3
        elif value & 0b11000000 == 0b11000000:
            expected_length = 2

        data = bytearray(first_byte)
        deadline = time.monotonic() + 0.1
        while len(data) < expected_length and time.monotonic() < deadline:
            if not select.select([fd], [], [], 0.01)[0]:
                continue
            chunk = os.read(fd, expected_length - len(data))
            if not chunk:
                break
            data.extend(chunk)
        return data.decode("utf-8", errors="ignore")

    first = "\x1b"
    if first != "\x1b":
        return first

    sequence = first
    deadline = time.monotonic() + 0.65

    while time.monotonic() < deadline:
        if select.select([fd], [], [], 0.04)[0]:
            sequence += os.read(fd, 1).decode("utf-8", errors="ignore")
            break
    if len(sequence) == 1:
        return first

    while time.monotonic() < deadline:
        if select.select([fd], [], [], 0.12)[0]:
            char = os.read(fd, 1).decode("utf-8", errors="ignore")
            if not char:
                break
            sequence += char
            if char.isalpha() or char == "~":
                break
            continue
        break

    return sequence


def _is_up_key(key: str) -> bool:
    return key in ("k",) or (key.startswith("\x1b") and key.endswith("A"))


def _is_down_key(key: str) -> bool:
    return key in ("j",) or (key.startswith("\x1b") and key.endswith("B"))


def _drain_pending_escape(
    pending_escape: str,
    pending_started_at: float | None,
) -> tuple[str | None, str, float | None]:
    if not pending_escape or pending_started_at is None:
        return None, pending_escape, pending_started_at
    if time.monotonic() - pending_started_at < 0.35:
        return None, pending_escape, pending_started_at
    return pending_escape, "", None


def _merge_escape_key(
    raw_key: str,
    pending_escape: str,
    pending_started_at: float | None,
) -> tuple[str | None, str, float | None]:
    if not pending_escape:
        if raw_key == "\x1b":
            return None, raw_key, time.monotonic()
        return raw_key, "", None

    combined = pending_escape + raw_key
    if pending_escape == "\x1b":
        if raw_key in ("[", "O"):
            return None, combined, pending_started_at
        return raw_key, "", None

    if pending_escape.startswith(("\x1b[", "\x1bO")):
        if combined[-1].isalpha() or combined[-1] == "~":
            return combined, "", None
        return None, combined, pending_started_at

    return raw_key, "", None


def _render_input_state(
    buffer: str,
    *,
    multiline: bool,
    selected_command: str | None = None,
) -> object:
    normalized = buffer.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if normalized.endswith("\n"):
        lines.append("")
    command_rows = _matching_slash_commands(normalized) if not multiline else []
    return build_input_box_lines(
        lines,
        command_rows=command_rows,
        selected_command=selected_command,
        placeholder="Type a message or / for commands",
        multiline=multiline,
    )


def _current_command_selection(
    buffer: str,
    current_selection: str | None,
    *,
    multiline: bool,
) -> tuple[list[tuple[str, str]], str | None]:
    if multiline:
        return [], None
    matches = _matching_slash_commands(buffer)
    if not matches:
        return [], None
    names = [name for name, _ in matches]
    if current_selection in names:
        return matches, current_selection
    return matches, names[0]


def _read_live_boxed_message() -> str:
    buffer = ""
    multiline = False
    selected_command: str | None = None
    pending_escape = ""
    pending_escape_started_at: float | None = None

    with _raw_stdin():
        renderer = _InlineTerminalRenderer(sys.stderr)
        try:
            matches, selected_command = _current_command_selection(
                buffer,
                selected_command,
                multiline=multiline,
            )
            renderer.render(
                _render_input_state(
                    buffer,
                    multiline=multiline,
                    selected_command=selected_command,
                )
            )
            while True:
                pending_key, pending_escape, pending_escape_started_at = _drain_pending_escape(
                    pending_escape,
                    pending_escape_started_at,
                )
                matches, selected_command = _current_command_selection(
                    buffer,
                    selected_command,
                    multiline=multiline,
                )
                if pending_key is None:
                    renderer.wait_for_input(
                        _render_input_state(
                            buffer,
                            multiline=multiline,
                            selected_command=selected_command,
                        )
                    )
                    raw_key = _read_key()
                    key, pending_escape, pending_escape_started_at = _merge_escape_key(
                        raw_key,
                        pending_escape,
                        pending_escape_started_at,
                    )
                    if key is None:
                        continue
                else:
                    key = pending_key

                if _is_up_key(key) and matches:
                    names = [name for name, _ in matches]
                    index = names.index(selected_command) if selected_command in names else 0
                    selected_command = names[(index - 1) % len(names)]
                    renderer.render(
                        _render_input_state(
                            buffer,
                            multiline=multiline,
                            selected_command=selected_command,
                        )
                    )
                    continue

                if _is_down_key(key) and matches:
                    names = [name for name, _ in matches]
                    index = names.index(selected_command) if selected_command in names else 0
                    selected_command = names[(index + 1) % len(names)]
                    renderer.render(
                        _render_input_state(
                            buffer,
                            multiline=multiline,
                            selected_command=selected_command,
                        )
                    )
                    continue

                if key in ("\r", "\n"):
                    if not multiline and matches and selected_command is not None:
                        buffer = selected_command
                        break
                    current_line = buffer.split("\n")[-1]
                    if multiline:
                        if current_line.strip() == MULTILINE_SENTINEL:
                            parts = buffer.split("\n")
                            buffer = "\n".join(parts[:-1]).strip()
                            break
                        buffer += "\n"
                        renderer.render(
                            _render_input_state(
                                buffer,
                                multiline=multiline,
                                selected_command=None,
                            )
                        )
                        continue

                    if buffer.strip() == MULTILINE_SENTINEL:
                        buffer = ""
                        multiline = True
                        selected_command = None
                        renderer.render(
                            _render_input_state(
                                buffer,
                                multiline=multiline,
                                selected_command=None,
                            )
                        )
                        continue

                    if current_line.endswith("\\"):
                        buffer = buffer[:-1].rstrip() + "\n"
                        multiline = True
                        selected_command = None
                        renderer.render(
                            _render_input_state(
                                buffer,
                                multiline=multiline,
                                selected_command=None,
                            )
                        )
                        continue

                    buffer = buffer.strip()
                    break

                if key in ("\x7f", "\b"):
                    if buffer:
                        buffer = buffer[:-1]
                        renderer.render(
                            _render_input_state(
                                buffer,
                                multiline=multiline,
                                selected_command=selected_command,
                            )
                        )
                    continue

                if key == "\x03":
                    raise KeyboardInterrupt

                if key == "\x04":
                    if not buffer:
                        raise EOFError
                    continue

                if key.startswith("\x1b"):
                    continue

                if matches and key in ("A", "B", "C", "D", "[", "O"):
                    continue

                if key == "\t":
                    if matches and selected_command is not None:
                        buffer = selected_command
                    else:
                        buffer += "    "
                    renderer.render(
                        _render_input_state(
                            buffer,
                            multiline=multiline,
                            selected_command=selected_command,
                        )
                    )
                    continue

                if key.isprintable():
                    buffer += key
                    matches, selected_command = _current_command_selection(
                        buffer,
                        selected_command,
                        multiline=multiline,
                    )
                    renderer.render(
                        _render_input_state(
                            buffer,
                            multiline=multiline,
                            selected_command=selected_command,
                        )
                    )
        finally:
            renderer.finish()

    err_console.print()
    return buffer


def _select_effort(current: str, *, supported: bool) -> str:
    if not sys.stdin.isatty() or not err_console.is_terminal:
        return current

    options = list(EFFORT_LEVELS)
    index = options.index(current)
    pending_escape = ""
    pending_escape_started_at: float | None = None

    with _raw_stdin():
        renderer = _InlineTerminalRenderer(sys.stderr)
        try:
            renderer.render(build_effort_menu_lines(current, options[index], supported=supported))
            while True:
                pending_key, pending_escape, pending_escape_started_at = _drain_pending_escape(
                    pending_escape,
                    pending_escape_started_at,
                )
                if pending_key is None:
                    renderer.wait_for_input(
                        build_effort_menu_lines(current, options[index], supported=supported)
                    )
                    raw_key = _read_key()
                    key, pending_escape, pending_escape_started_at = _merge_escape_key(
                        raw_key,
                        pending_escape,
                        pending_escape_started_at,
                    )
                    if key is None:
                        continue
                else:
                    key = pending_key
                if key in ("\r", "\n"):
                    break
                if key in ("\x03",):
                    raise KeyboardInterrupt
                if key == "\x1b":
                    renderer.finish()
                    err_console.print()
                    return current
                if _is_up_key(key):
                    index = (index - 1) % len(options)
                elif _is_down_key(key):
                    index = (index + 1) % len(options)
                elif key in ("A", "B", "C", "D", "[", "O"):
                    continue
                elif key in ("1", "2", "3", "4"):
                    index = int(key) - 1
                renderer.render(build_effort_menu_lines(current, options[index], supported=supported))
        finally:
            renderer.finish()

    err_console.print()
    return options[index]


class _InlineTerminalRenderer(BottomTerminalRenderer):
    def __init__(self, stream) -> None:
        super().__init__(stream, clear_on_finish=True)


class _ReasoningParser:
    def __init__(self) -> None:
        self.reasoning_text = ""
        self.in_reasoning = False
        self.saw_reasoning = False
        self._emitted_any_visible = False
        self._buffer = ""
        self._implicit_reasoning = False

    def feed(self, chunk: str) -> tuple[str, str, bool]:
        self._buffer += chunk
        visible_parts: list[str] = []
        reasoning_parts: list[str] = []
        just_closed = False

        while self._buffer:
            if self.in_reasoning:
                if self._implicit_reasoning and _should_end_implicit_reasoning(self.reasoning_text, self._buffer):
                    self.in_reasoning = False
                    self._implicit_reasoning = False
                    just_closed = True
                    continue
                if self._buffer.startswith(CLOSE_THINK_TAG):
                    self._buffer = self._buffer[len(CLOSE_THINK_TAG):]
                    self.in_reasoning = False
                    self._implicit_reasoning = False
                    just_closed = True
                    continue
                if _starts_with_partial_tag(self._buffer, CLOSE_THINK_TAG):
                    break
                char = self._buffer[0]
                reasoning_parts.append(char)
                self.reasoning_text += char
                self._buffer = self._buffer[1:]
                continue

            if self._buffer.startswith(OPEN_THINK_TAG):
                self.saw_reasoning = True
                self.in_reasoning = True
                self._implicit_reasoning = False
                self._buffer = self._buffer[len(OPEN_THINK_TAG):]
                continue

            close_index = self._buffer.find(CLOSE_THINK_TAG)
            open_index = self._buffer.find(OPEN_THINK_TAG)
            if (
                not self.saw_reasoning
                and not self._emitted_any_visible
                and close_index != -1
                and (open_index == -1 or close_index < open_index)
            ):
                prefix = self._buffer[:close_index]
                if prefix:
                    reasoning_parts.append(prefix)
                    self.reasoning_text += prefix
                self.saw_reasoning = True
                self._buffer = self._buffer[close_index + len(CLOSE_THINK_TAG):]
                just_closed = True
                continue

            if _starts_with_partial_tag(self._buffer, OPEN_THINK_TAG):
                break
            if (
                not self.saw_reasoning
                and not self._emitted_any_visible
                and _starts_with_partial_tag(self._buffer, CLOSE_THINK_TAG)
            ):
                break
            if (
                not self.saw_reasoning
                and not self._emitted_any_visible
                and _looks_like_reasoning_prefix(self._buffer)
            ):
                self.saw_reasoning = True
                self.in_reasoning = True
                self._implicit_reasoning = True
                continue
            if (
                not self.saw_reasoning
                and not self._emitted_any_visible
                and _might_be_reasoning_prefix(self._buffer)
            ):
                break

            char = self._buffer[0]
            visible_parts.append(char)
            self._buffer = self._buffer[1:]
            if not char.isspace():
                self._emitted_any_visible = True

        return "".join(visible_parts), "".join(reasoning_parts), just_closed


def _extract_visible_text(text: str) -> tuple[str, str, bool]:
    visible: list[str] = []
    reasoning: list[str] = []
    in_reasoning = False
    i = 0

    while i < len(text):
        remaining = text[i:]
        if not in_reasoning and remaining.startswith(OPEN_THINK_TAG):
            in_reasoning = True
            i += len(OPEN_THINK_TAG)
            continue
        if in_reasoning and remaining.startswith(CLOSE_THINK_TAG):
            in_reasoning = False
            i += len(CLOSE_THINK_TAG)
            continue

        if len(remaining) < len(OPEN_THINK_TAG) and OPEN_THINK_TAG.startswith(remaining):
            break
        if len(remaining) < len(CLOSE_THINK_TAG) and CLOSE_THINK_TAG.startswith(remaining):
            break

        if in_reasoning:
            reasoning.append(text[i])
        else:
            visible.append(text[i])
        i += 1

    return "".join(visible), "".join(reasoning), in_reasoning


def _looks_like_reasoning_prefix(text: str) -> bool:
    sample = text.lstrip()
    if not sample:
        return False
    lowered = sample.lower()
    if any(marker in lowered for marker in REASONING_PREFIX_CANDIDATES):
        return True
    if sample.startswith("1.") and "\n" in sample:
        return True
    return False


def _starts_with_partial_tag(text: str, tag: str) -> bool:
    if not text or len(text) >= len(tag):
        return False
    return tag.startswith(text)


def _might_be_reasoning_prefix(text: str) -> bool:
    sample = text.lstrip().lower()
    if not sample:
        return False
    if len(sample) > 64:
        return False
    return any(candidate.startswith(sample) for candidate in REASONING_PREFIX_CANDIDATES)


def _should_end_implicit_reasoning(reasoning_text: str, buffer: str) -> bool:
    if not reasoning_text or not (reasoning_text.endswith("\n") or reasoning_text.endswith("\r")):
        return False
    sample = buffer.lstrip()
    if not sample:
        return False
    if _looks_like_reasoning_line_start(sample) or _might_be_reasoning_line_start(sample):
        return False
    return True


def _looks_like_reasoning_line_start(text: str) -> bool:
    sample = text.lstrip()
    lowered = sample.lower()
    if sample.startswith(("**", "* ", "- ", "• ", "> ")):
        return True
    if re.match(r"^\d+[\.\)]\s", sample):
        return True
    markers = (
        "input:",
        "language:",
        "meaning:",
        "intent:",
        "option ",
        "analysis",
        "reasoning",
        "determine",
        "draft",
        "final",
        "review",
        "self-correction",
        "let's",
        "the user",
        "i need to",
        "i should",
        "first",
        "however",
        "summary",
        "wait",
        "check",
        "步骤",
        "分析",
        "输入：",
        "输入:",
        "意图",
        "最终",
        "好的，用户",
        "用户发来",
        "用户可能",
        "看起来",
        "我需要",
        "我应该",
        "首先",
        "不过",
        "另外",
        "同时",
        "现在需要",
        "根据我的知识库",
        "总结",
        "总结：",
    )
    return any(lowered.startswith(marker) for marker in markers)


def _might_be_reasoning_line_start(text: str) -> bool:
    sample = text.lstrip().lower()
    if not sample:
        return False
    if len(sample) > 48:
        return False
    candidates = (
        "input:",
        "language:",
        "meaning:",
        "intent:",
        "option ",
        "analysis",
        "reasoning",
        "determine",
        "draft",
        "final",
        "review",
        "self-correction",
        "let's",
        "the user",
        "i need to",
        "i should",
        "first",
        "however",
        "summary",
        "wait",
        "check",
        "步骤",
        "分析",
        "输入：",
        "输入:",
        "意图",
        "最终",
        "好的，用户",
        "用户发来",
        "用户可能",
        "看起来",
        "我需要",
        "我应该",
        "首先",
        "不过",
        "另外",
        "同时",
        "现在需要",
        "根据我的知识库",
        "总结",
        "总结：",
        "* ",
        "- ",
        "• ",
        "> ",
    )
    if any(candidate.startswith(sample) for candidate in candidates):
        return True
    return bool(re.match(r"^\d+[\.\)]?$", sample))


def _stream_response(
    model,
    tokenizer,
    messages,
    *,
    model_label,
    temp,
    top_p,
    max_tokens,
    verbose,
    chat_template_kwargs: dict | None = None,
):
    """Stream a response, collecting the full text. Returns (full_text, last_response, reasoning_text)."""
    full_text = ""
    last_response = None
    wrote_output = False
    stream_to_stdout = not err_console.is_terminal
    parser = _ReasoningParser()
    prev_reasoning = False

    with daydreaming_status(err_console, model_label) as status:
        for response in engine.generate_stream(
            model, tokenizer, messages,
            max_tokens=max_tokens, temp=temp, top_p=top_p,
            chat_template_kwargs=chat_template_kwargs,
        ):
            raw_chunk = response.text or ""
            text_chunk, reasoning_chunk, reasoning_closed = parser.feed(raw_chunk)

            # Reasoning state transitions → drive the status display
            if parser.in_reasoning and not prev_reasoning:
                status.start_reasoning()
            if reasoning_chunk:
                status.append_reasoning(reasoning_chunk)
            if reasoning_closed:
                status.end_reasoning()
            prev_reasoning = parser.in_reasoning

            # Stats updates
            if response.prompt_tps and not wrote_output:
                status.update(phase="prefill", tokens_per_second=response.prompt_tps)

            # First visible token → switch from animation to output mode
            if text_chunk and not wrote_output:
                status.ensure_minimum_wait(0.28)
                status.update(waiting=False, phase=None)

            full_text += text_chunk
            last_response = response

            if text_chunk:
                if stream_to_stdout:
                    print(text_chunk, end="", flush=True)
                else:
                    status.append_output(text_chunk)
                wrote_output = True
                if response.generation_tps:
                    status.update(phase=None, tokens_per_second=response.generation_tps)
            elif response.generation_tps:
                status.update(tokens_per_second=response.generation_tps)

            if response.finish_reason:
                break

    if wrote_output and stream_to_stdout:
        print()

    if verbose and last_response:
        _print_stats(last_response)

    return full_text, last_response, parser.reasoning_text


def _select_dreaming_mode() -> str | None:
    """Arrow-key menu to choose REMing or Daydream.

    Pattern: copy _select_effort() structure exactly.
    Returns selected mode string or None on Esc.
    """
    if not sys.stdin.isatty() or not err_console.is_terminal:
        return None

    options = ["reming", "daydream"]
    index = 0
    pending_escape = ""
    pending_escape_started_at: float | None = None

    with _raw_stdin():
        renderer = _InlineTerminalRenderer(sys.stderr)
        try:
            renderer.render(build_dreaming_menu_lines(options[index]))
            while True:
                pending_key, pending_escape, pending_escape_started_at = _drain_pending_escape(
                    pending_escape,
                    pending_escape_started_at,
                )
                if pending_key is None:
                    renderer.wait_for_input(
                        build_dreaming_menu_lines(options[index])
                    )
                    raw_key = _read_key()
                    key, pending_escape, pending_escape_started_at = _merge_escape_key(
                        raw_key,
                        pending_escape,
                        pending_escape_started_at,
                    )
                    if key is None:
                        continue
                else:
                    key = pending_key
                if key in ("\r", "\n"):
                    break
                if key in ("\x03",):
                    raise KeyboardInterrupt
                if key == "\x1b":
                    renderer.finish()
                    err_console.print()
                    return None
                if _is_up_key(key):
                    index = (index - 1) % len(options)
                elif _is_down_key(key):
                    index = (index + 1) % len(options)
                elif key in ("A", "B", "C", "D", "[", "O"):
                    continue
                renderer.render(build_dreaming_menu_lines(options[index]))
        finally:
            renderer.finish()

    err_console.print()
    return options[index]


def _select_session(sessions: list) -> object | None:
    """Arrow-key menu to choose a session to resume.

    Pattern: copy _select_effort() structure exactly.
    Returns selected ChatSession or None on Esc.
    """
    if not sys.stdin.isatty() or not err_console.is_terminal:
        return None

    if not sessions:
        return None

    index = 0
    pending_escape = ""
    pending_escape_started_at: float | None = None

    with _raw_stdin():
        renderer = _InlineTerminalRenderer(sys.stderr)
        try:
            window = _session_window(sessions, index)
            renderer.render(
                build_session_list_lines(
                    window,
                    index - _session_window_start(sessions, index),
                )
            )
            while True:
                pending_key, pending_escape, pending_escape_started_at = _drain_pending_escape(
                    pending_escape,
                    pending_escape_started_at,
                )
                if pending_key is None:
                    window = _session_window(sessions, index)
                    renderer.wait_for_input(
                        build_session_list_lines(
                            window,
                            index - _session_window_start(sessions, index),
                        )
                    )
                    raw_key = _read_key()
                    key, pending_escape, pending_escape_started_at = _merge_escape_key(
                        raw_key,
                        pending_escape,
                        pending_escape_started_at,
                    )
                    if key is None:
                        continue
                else:
                    key = pending_key
                if key in ("\r", "\n"):
                    break
                if key in ("\x03",):
                    raise KeyboardInterrupt
                if key == "\x1b":
                    renderer.finish()
                    err_console.print()
                    return None
                if _is_up_key(key):
                    index = (index - 1) % len(sessions)
                elif _is_down_key(key):
                    index = (index + 1) % len(sessions)
                elif key in ("A", "B", "C", "D", "[", "O"):
                    continue
                window = _session_window(sessions, index)
                renderer.render(
                    build_session_list_lines(
                        window,
                        index - _session_window_start(sessions, index),
                    )
                )
        finally:
            renderer.finish()

    err_console.print()
    return sessions[index]


def _session_window_start(sessions: list, index: int, *, window_size: int = 10) -> int:
    if len(sessions) <= window_size:
        return 0
    start = max(0, index - (window_size // 2))
    return min(start, len(sessions) - window_size)


def _session_window(sessions: list, index: int, *, window_size: int = 10) -> list:
    start = _session_window_start(sessions, index, window_size=window_size)
    return sessions[start:start + window_size]


def _run_dreaming(
    model,
    tokenizer,
    messages,
    session,
    mode,
    *,
    model_label,
    session_memories: list[Memory],
    temp,
    max_tokens,
) -> list:
    """Execute the selected dreaming mode with animated status.

    1. Create DreamingStatus instance
    2. Convert messages to ChatMessage list
    3. Call run_reming() or run_dreaming() from dreaming.py
       with on_phase/on_token callbacks that update DreamingStatus
    4. Stop status, return memories
    """
    from daydream.utils import dreaming_status as _dreaming_status_ctx

    # Convert messages dicts to ChatMessage objects
    chat_messages = [
        ChatMessage(
            role=m["role"],
            content=m["content"],
            timestamp=time.time(),
        )
        for m in messages
    ]

    session_id = session.session_id if session is not None else ""

    with _dreaming_status_ctx(err_console, model_label) as status:
        if mode == "daydream":
            total = 3
            phase_num = [0]

            def on_phase(name: str, text: str) -> None:
                phase_num[0] += 1
                status.set_phase(name, text, number=phase_num[0], total=total)

            def on_token(text: str) -> None:
                status.append_text(text)

            memories = run_dreaming(
                model, tokenizer, chat_messages, session_id,
                existing_memories=session_memories,
                temp=temp, max_tokens=max_tokens,
                on_phase=on_phase, on_token=on_token,
            )
        else:
            def on_phase(name: str, text: str) -> None:
                status.set_phase(name, text, number=1, total=1)

            def on_token(text: str) -> None:
                status.append_text(text)

            memories = run_reming(
                model, tokenizer, chat_messages,
                temp=temp, max_tokens=max_tokens,
                on_phase=on_phase, on_token=on_token,
            )

    # Tag memories with session_id if we have one
    if session is not None:
        for mem in memories:
            mem.session_id = session.session_id

    return memories


def run_oneshot(
    model_name: str,
    *,
    prompt: str | None = None,
    temp: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 4096,
    system: str | None = None,
    verbose: bool = False,
    initial_effort: str = "default",
    display_name: str | None = None,
) -> None:
    """Run a single generation and exit."""
    # Read from stdin if no prompt given
    if prompt is None:
        if sys.stdin.isatty():
            err_console.print("[red]Error:[/] No prompt provided.")
            raise SystemExit(1)
        prompt = sys.stdin.read().strip()
        if not prompt:
            err_console.print("[red]Error:[/] Empty input.")
            raise SystemExit(1)

    resolved_name = ensure_runtime_model(model_name, auto_pull=True, register_alias=True)
    with err_console.status("Preparing model..."):
        model, tokenizer = engine.load_model(resolved_name, ensure_available=False)

    base_messages = [{"role": "user", "content": prompt}]
    request_messages = _build_request_messages(
        base_messages,
        system_prompt=system,
        effort=initial_effort,
        model_name=resolved_name,
    )
    chat_template_kwargs = _effort_chat_template_kwargs(initial_effort, tokenizer)

    _stream_response(
        model, tokenizer, request_messages,
        model_label=display_name or reverse_lookup(resolved_name) or model_name,
        temp=temp, top_p=top_p, max_tokens=max_tokens, verbose=verbose,
        chat_template_kwargs=chat_template_kwargs,
    )


def run_chat(
    model_name: str,
    *,
    temp: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 4096,
    system: str | None = None,
    verbose: bool = False,
    initial_effort: str = "default",
    display_name: str | None = None,
) -> None:
    """Run an interactive chat REPL."""
    repo_id = ensure_runtime_model(model_name, auto_pull=True, register_alias=True)
    with err_console.status("Preparing model..."):
        model, tokenizer = engine.load_model(repo_id, ensure_available=False)
    short = display_name or reverse_lookup(repo_id) or model_name

    err_console.print(f"[bold cyan]{short}[/]")
    err_console.print("[dim]Use / for commands.[/dim]")
    err_console.print()

    messages: list[dict] = []
    last_reasoning = ""
    effort = initial_effort if initial_effort in EFFORT_LEVELS else "default"
    effort_supported = _model_supports_effort(repo_id)

    # Persistence state
    session: ChatSession | None = None  # None = memoryless mode (default)
    session_memories: list[Memory] = []

    while True:
        try:
            user_input = _read_boxed_message()
        except (EOFError, KeyboardInterrupt):
            err_console.print()
            break
        stripped = user_input.strip()
        if not stripped:
            continue

        # Commands
        if stripped in ("/quit", "/exit", "/q"):
            break
        if stripped in ("/reset", "/clear", "/r"):
            messages = []
            if session is not None:
                session.messages = []
                session.updated_at = time.time()
                save_session(session)
            err_console.print("[dim]Chat history cleared.[/dim]")
            continue
        if stripped in ("/help", "/h", "/?"):
            err_console.print(f"[dim]{MULTILINE_SENTINEL}      — start/end multiline input[/dim]")
            err_console.print("[dim]\\        — continue input on the next line[/dim]")
            err_console.print("[dim]/new      — start a persistent chat session[/dim]")
            err_console.print("[dim]/resume   — resume a saved session[/dim]")
            err_console.print("[dim]/dreaming — consolidate memories from conversation[/dim]")
            err_console.print("[dim]/memory   — view extracted memories[/dim]")
            err_console.print("[dim]/effort   — pick instant / short / default / long[/dim]")
            err_console.print("[dim]/reset    — clear conversation history[/dim]")
            err_console.print("[dim]/clear    — alias for /reset[/dim]")
            err_console.print("[dim]/t        — show last captured reasoning[/dim]")
            err_console.print("[dim]/quit     — exit chat[/dim]")
            err_console.print("[dim]/help     — show this help[/dim]")
            continue
        if stripped == "/":
            err_console.print("[dim]Type /effort, /reset, /clear, /t, /help, or /quit.[/dim]")
            continue
        if stripped.startswith("/effort"):
            parts = stripped.split(maxsplit=1)
            selected = _normalize_effort(parts[1]) if len(parts) > 1 else None
            if selected is None:
                selected = _select_effort(effort, supported=effort_supported)
            effort = selected
            note = "[dim]Model may ignore this setting.[/dim]" if not effort_supported else ""
            err_console.print(f"[dim]Reasoning effort set to {effort}.[/dim] {note}".rstrip())
            continue
        if stripped in ("/t", "/thoughts"):
            err_console.print(render_expanded_reasoning(last_reasoning))
            continue
        if stripped == "/new":
            import uuid
            session = ChatSession(
                session_id=uuid.uuid4().hex,
                model=short,
                title="",
                created_at=time.time(),
                updated_at=time.time(),
                messages=[],
                memories=[],
            )
            session_memories = []
            # Copy existing messages into session if any
            for msg in messages:
                session.messages.append(ChatMessage(
                    role=msg["role"], content=msg["content"], timestamp=time.time(),
                ))
            save_session(session)
            err_console.print(f"[dim]Persistent session started ({session.session_id[:8]}...)[/dim]")
            continue
        if stripped == "/resume":
            available_sessions = list_sessions()
            if not available_sessions:
                err_console.print("[dim]No saved sessions.[/dim]")
                continue
            selected_session = _select_session(available_sessions)
            if selected_session is None:
                continue
            session = selected_session
            messages = [{"role": m.role, "content": m.content} for m in session.messages]
            session_memories = load_memories(session.session_id)
            err_console.print(f"[dim]Resumed: {session.title} ({len(messages)} messages)[/dim]")
            continue
        if stripped == "/dreaming":
            if session is None:
                err_console.print("[dim]Start a persistent session with /new before dreaming.[/dim]")
                continue
            if len(messages) < 2:
                err_console.print("[dim]Need at least one exchange to dream about.[/dim]")
                continue
            mode = _select_dreaming_mode()
            if mode is None:
                continue
            new_memories = _run_dreaming(
                model, tokenizer, messages, session, mode,
                model_label=short,
                session_memories=session_memories,
                temp=0.3,
                max_tokens=2048,
            )
            if not new_memories:
                err_console.print("[dim]No memories extracted.[/dim]")
                continue
            for mem_line in build_memory_display_lines(new_memories):
                err_console.print(mem_line)
            if _confirm_memory_import(new_memories):
                session_memories.extend(new_memories)
                session.memories = session_memories
                session.updated_at = time.time()
                save_memories(session.session_id, session_memories)
                save_session(session)
                err_console.print(f"[dim]Added {len(new_memories)} memories to this session.[/dim]")
            else:
                err_console.print("[dim]Discarded extracted memories.[/dim]")
            continue
        if stripped == "/memory":
            if session is None:
                err_console.print("[dim]No session memory is active. Start with /new or /resume.[/dim]")
                continue
            if not session_memories:
                err_console.print("[dim]No memories yet. Use /dreaming to extract memories.[/dim]")
                continue
            mem_lines = build_memory_display_lines(session_memories)
            for mem_line in mem_lines:
                err_console.print(mem_line)
            continue

        messages.append({"role": "user", "content": user_input})
        request_messages = _build_request_messages(
            messages,
            system_prompt=system,
            effort=effort,
            model_name=repo_id,
            session_memories=session_memories if session is not None else None,
        )
        chat_template_kwargs = _effort_chat_template_kwargs(effort, tokenizer)

        full_text, _, reasoning_text = _stream_response(
            model, tokenizer, request_messages,
            model_label=short,
            temp=temp, top_p=top_p, max_tokens=max_tokens, verbose=verbose,
            chat_template_kwargs=chat_template_kwargs,
        )
        last_reasoning = reasoning_text

        messages.append({"role": "assistant", "content": full_text})

        if session is not None:
            session.messages.append(ChatMessage(
                role="user", content=user_input, timestamp=time.time(),
                reasoning="",
            ))
            session.messages.append(ChatMessage(
                role="assistant", content=full_text, timestamp=time.time(),
                reasoning=last_reasoning,
            ))
            session.updated_at = time.time()
            if not session.title:
                session.title = user_input[:50].strip()
            save_session(session)

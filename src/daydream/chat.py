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
from dataclasses import dataclass
from typing import Callable

from rich.console import Console, Group
from rich.text import Text

from daydream import engine
from daydream.config import get_default_cli_page_mode, set_default_cli_page_mode
from daydream.models import ensure_runtime_model, is_model_available_locally, pull_model
from daydream.registry import reverse_lookup
from daydream.speculative import (
    default_draft_for_model,
    default_num_draft_tokens,
    draft_model_for_family,
)
from daydream.utils import (
    BottomTerminalRenderer,
    CLI_PAGE_MODES,
    InlineFlowRenderer,
    build_cli_page_menu_lines,
    build_draft_menu_lines,
    build_dreaming_menu_lines,
    build_effort_menu_lines,
    build_input_box_lines,
    build_memory_display_lines,
    build_session_list_lines,
    daydreaming_status,
    measure_renderable_lines,
    render_reasoning_line,
    render_reasoning_box,
    render_expanded_reasoning,
    render_input_box,
    render_status_footer,
)
from daydream.storage import (
    ChatSession,
    ChatMessage,
    Memory,
    delete_session,
    rename_session,
    save_session,
    list_sessions,
    save_memories,
    load_memories,
)
from daydream.dreaming import run_reming, run_dreaming

console = Console()
err_console = Console(stderr=True)

MULTILINE_SENTINEL = '"""'
_PENDING_ESCAPE_TIMEOUT = 0.03
OPEN_THINK_TAG = "<think>"
CLOSE_THINK_TAG = "</think>"
EFFORT_LEVELS = ("instant", "short", "default", "long")
CLI_PAGE_LEVELS = CLI_PAGE_MODES
_DRAFT_COMMAND_LEVELS = ("on", "off")
DRAFT_SLASH_COMMAND = "/draft(beta)"
SLASH_COMMANDS = (
    ("/effort", "adjust reasoning depth"),
    (DRAFT_SLASH_COMMAND, "toggle draft acceleration for supported models"),
    ("/cli-page", "choose loose or tight page spacing"),
    ("/help", "show available commands"),
    ("/new", "start a persistent chat session"),
    ("/resume", "resume a saved session"),
    ("/forget", "delete a saved session"),
    ("/rename", "rename a saved session"),
    ("/dreaming", "consolidate memories from conversation"),
    ("/memory", "view extracted memories"),
    ("/reset", "clear conversation history"),
    ("/clear", "alias for /reset"),
    ("/t", "expand the last reasoning trace"),
    ("/quit", "exit chat"),
)
REASONING_PREFIX_CANDIDATES = (
    "usually, this means",
    "this implies",
    "user says",
    "intent is",
    "response should",
    "keep it short",
    "let's check the system instruction again",
    "let's check the instruction again",
    "the instruction says",
    "now the actual response",
    "now the actual answer",
    "thinking process",
    "thinking process:",
    "thinking block",
    "thinking block:",
    "revised plan",
    "revised plan:",
    "final answer:",
    "thinking:",
    "actually, looking at",
    "let's just respond",
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
    "response:",
    "constraint check:",
    "thinking block lines:",
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
EMPHASIZED_REASONING_PREFIX_CANDIDATES = (
    "*simpler",
    "*friendly",
    "*let's",
    "*final",
    "*wait",
    "*okay",
    "*draft",
    "*review",
    "*analyze",
    "*determine",
    "*option",
    "*self-correction",
)

_IMPLICIT_REASONING_MAX_CHARS = 8000

_META_REASONING_PREFIX_CANDIDATES = (
    "now the actual response",
    "now the actual answer",
    "now the response",
    "now the answer",
)
_FINAL_ANSWER_LABELS = (
    "final answer:",
    "final answer：",
    "answer:",
    "answer：",
    "response:",
    "response：",
    "final output:",
    "final output：",
    "回答:",
    "回答：",
    "答案:",
    "答案：",
    "回复:",
    "回复：",
)


@dataclass
class _TranscriptBlock:
    kind: str
    content: str = ""
    elapsed: float | None = None


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
    cli_page_mode: str = "loose",
) -> str:
    if input_func is input and sys.stdin.isatty() and err_console.is_terminal:
        return _read_live_boxed_message(cli_page_mode=cli_page_mode)

    err_console.print(render_input_box([""], placeholder="Type a message or / for commands"))
    try:
        first_line = input_func("│ ")
        return _collect_multiline_message(first_line, lambda _: input_func("│ "))
    finally:
        _print_gap(cli_page_mode)


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


def _normalize_cli_page_mode(value: str) -> str | None:
    choice = value.strip().lower()
    if choice in CLI_PAGE_LEVELS:
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
    # Detect thinking support: check has_thinking attribute first,
    # then fall back to inspecting the Jinja template for enable_thinking.
    # Distilled models (e.g. Qwen3.5-Claude) may lack has_thinking but
    # still have a template that supports enable_thinking.
    has_thinking = bool(getattr(tokenizer, "has_thinking", False))
    if not has_thinking:
        template = getattr(tokenizer, "chat_template", None)
        if template and "enable_thinking" in str(template):
            has_thinking = True
    if not has_thinking:
        return {}
    # Always enable thinking so the model wraps reasoning in
    # <think>...</think> tags instead of leaking it into visible text.
    # Always pass thinking_budget — Jinja templates silently ignore
    # unknown variables, and omitting it can cause some templates to
    # default to a very small budget (cutting reasoning short).
    if effort == "default":
        return {"enable_thinking": True, "thinking_budget": 10000}
    if effort == "long":
        return {"enable_thinking": True, "thinking_budget": 10000}
    if effort == "instant":
        return {"enable_thinking": False}
    if effort == "short":
        return {"enable_thinking": True, "thinking_budget": 200}
    return {"enable_thinking": True, "thinking_budget": 10000}


def _effort_system_prompt(effort: str, model_name: str) -> str | None:
    if effort == "default" or not _model_supports_effort(model_name):
        return None

    prompts = {
        "instant": (
            "Answer directly and concisely. Prefer minimal internal reasoning. "
            "Skip preamble and go straight to the answer."
        ),
        "short": (
            "Reason briefly and efficiently before answering. "
            "Keep analysis minimal, skip obvious steps, and focus on the answer."
        ),
        "long": (
            "Reason thoroughly before answering. "
            "Explore key angles, consider edge cases, verify your logic, and then give the answer."
        ),
    }
    return prompts.get(effort)


def _normalize_draft_command(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in _DRAFT_COMMAND_LEVELS:
        return normalized
    return None


def _resolve_speculative_settings(
    *,
    model_name: str,
    resolved_model: str,
    draft_mode: str | None,
    just_downloaded_main: bool,
    allow_prompt: bool,
) -> tuple[str | None, int | None]:
    if draft_mode == "off":
        return None, None

    draft_model = draft_model_for_family(resolved_model)
    if draft_model is None:
        if draft_mode == "force":
            raise ValueError("Manual draft mode is only supported for Qwen3.5 MLX models.")
        return None, None

    num_draft_tokens = default_num_draft_tokens(resolved_model)
    if num_draft_tokens is None:
        return None, None

    if is_model_available_locally(draft_model):
        return draft_model, num_draft_tokens

    if draft_mode == "force":
        pull_model(draft_model, register_alias=True)
        return draft_model, num_draft_tokens

    return None, None


def _build_request_messages(
    history: list[dict],
    *,
    system_prompt: str | None,
    effort: str,
    model_name: str,
    session_memories: list[Memory] | None = None,
) -> list[dict]:
    request_messages: list[dict] = []
    system_sections: list[str] = []
    if system_prompt:
        system_sections.append(system_prompt.strip())
    memory_prompt = _build_session_memory_prompt(session_memories or [])
    if memory_prompt:
        system_sections.append(memory_prompt)
    effort_prompt = _effort_system_prompt(effort, model_name)
    if effort_prompt:
        system_sections.append(effort_prompt)
    if system_sections:
        request_messages.append({"role": "system", "content": "\n\n".join(system_sections)})
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
        "## Session Memory",
        "Session memory for this persistent chat only.",
        "The following are memories extracted from earlier exchanges in this persistent chat session.",
        "These represent knowledge, preferences, and patterns you have learned about the user.",
        "Use them naturally when relevant — do not list them back to the user unless asked.",
        "Treat high-importance memories (>0.7) as key context that should inform your responses.",
    ]
    for memory in list(unique.values())[:20]:
        content = memory.content.strip().replace("\n", " ")
        if len(content) > 200:
            content = content[:197] + "..."
        lines.append(f"- [{memory.category}|{memory.importance:.1f}] {content}")

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


def _confirm_session_delete(session: ChatSession) -> bool:
    if not sys.stdin.isatty():
        return True
    while True:
        err_console.print(
            f"[dim]Delete saved session '{session.title or '(untitled)'}'? (y/n)[/dim]"
        )
        reply = err_console.input("[dim]> [/dim]").strip().lower()
        if reply in ("y", "yes"):
            return True
        if reply in ("n", "no"):
            return False
        err_console.print("[dim]Please answer y or n.[/dim]")


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
        if select.select([fd], [], [], 0.015)[0]:
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
    if time.monotonic() - pending_started_at < _PENDING_ESCAPE_TIMEOUT:
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


def _wait_for_menu_input(
    renderer: _InlineTerminalRenderer | None,
    renderable: list[str] | Iterable[str] | object,
    *,
    pending_escape: str,
) -> bool:
    if pending_escape:
        if renderer is None or not hasattr(renderer, "_terminal_size") or not hasattr(renderer, "_last_size"):
            return True
        fd = sys.stdin.fileno()
        ready, _, _ = select.select([fd], [], [], 0.01)
        size = renderer._terminal_size()
        if size != renderer._last_size:
            renderer.render(renderable)
        return bool(ready)
    if renderer is not None:
        renderer.wait_for_input(renderable)
    return True


def _render_input_state(
    buffer: str,
    *,
    multiline: bool,
    selected_command: str | None = None,
    cli_page_mode: str = "loose",
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
        cli_page_mode=cli_page_mode,
    )


def _input_box_reserve_lines() -> int:
    return max(1, len(_render_input_state("", multiline=False)))


def _status_overlay_reserve_lines(model_label: str) -> int:
    color_system = getattr(err_console, "color_system", None)
    if not isinstance(color_system, str):
        color_system = "truecolor"
    preview = Group(
        render_reasoning_box("", 0, label="Daydreaming"),
        Text(),
        render_status_footer(
            model_label,
            hint="Use /t to expand reasoning",
        ),
    )
    return max(1, measure_renderable_lines(preview, color_system=color_system))


def _is_tight_cli_page_mode(cli_page_mode: str) -> bool:
    return str(cli_page_mode).strip().lower() == "tight"


def _print_gap(cli_page_mode: str, *, loose: int = 1, tight: int = 1) -> None:
    count = tight if _is_tight_cli_page_mode(cli_page_mode) else loose
    for _ in range(max(0, count)):
        err_console.print()


def _clear_visible_terminal() -> None:
    if not err_console.is_terminal:
        return
    stream = getattr(err_console, "file", sys.stderr)
    if not getattr(stream, "isatty", lambda: False)():
        return
    stream.write("\x1b[2J\x1b[H")
    stream.flush()


def _transcript_blocks_from_messages(
    messages: list[ChatMessage],
    *,
    limit_visible: int | None = None,
) -> list[_TranscriptBlock]:
    blocks: list[_TranscriptBlock] = []
    visible_messages = [message for message in messages if message.role in {"user", "assistant"}]
    if limit_visible is not None:
        visible_messages = visible_messages[-limit_visible:]
    for message in visible_messages:
        if message.role == "user":
            blocks.append(_TranscriptBlock("user", message.content))
            continue
        if message.role != "assistant":
            continue
        if message.reasoning.strip():
            blocks.append(_TranscriptBlock("reasoning_summary"))
        blocks.append(_TranscriptBlock("assistant", message.content))
    if blocks and blocks[-1].kind == "reasoning_hint":
        blocks.pop()
    return blocks


def _append_transcript_turn(
    blocks: list[_TranscriptBlock],
    *,
    user_input: str,
    assistant_text: str,
    reasoning_elapsed: float | None,
    had_reasoning: bool,
) -> None:
    while blocks and blocks[-1].kind == "reasoning_hint":
        blocks.pop()
    blocks.append(_TranscriptBlock("user", user_input))
    if had_reasoning:
        blocks.append(_TranscriptBlock("reasoning_summary", elapsed=reasoning_elapsed))
    if assistant_text:
        blocks.append(_TranscriptBlock("assistant", assistant_text))
    if had_reasoning and assistant_text:
        blocks.append(_TranscriptBlock("reasoning_hint", "Use /t to expand reasoning"))


def _repack_tight_page(model_label: str, blocks: list[_TranscriptBlock]) -> None:
    _clear_visible_terminal()
    err_console.print(f"[bold cyan]{model_label}[/]")
    err_console.print("[dim]Use / for commands.[/dim]")
    _print_gap("tight")

    for index, block in enumerate(blocks):
        if block.kind == "user":
            err_console.print("[bold green]You[/]")
            err_console.print(block.content, markup=False, highlight=False)
        elif block.kind == "assistant":
            err_console.print(f"[bold cyan]{model_label}[/]")
            err_console.print(block.content, markup=False, highlight=False)
        elif block.kind == "reasoning_summary":
            elapsed = block.elapsed if block.elapsed is not None else 0.0
            if elapsed > 0.0:
                err_console.print(render_reasoning_line(elapsed, active=False))
            else:
                err_console.print("[dim]▸ Daydreamed[/dim]")
        elif block.kind == "reasoning_hint":
            err_console.print(f"[dim]{block.content}[/dim]")
        else:
            continue

        if index < len(blocks) - 1:
            _print_gap("tight")

    if blocks:
        _print_gap("tight")


def _repack_loose_page(model_label: str, blocks: list[_TranscriptBlock]) -> None:
    visible_blocks = blocks[-30:]

    _clear_visible_terminal()
    err_console.print(f"[bold cyan]{model_label}[/]")
    err_console.print("[dim]Use / for commands.[/dim]")
    _print_gap("loose")

    for index, block in enumerate(visible_blocks):
        if block.kind == "user":
            err_console.print("[bold green]You[/]")
            err_console.print(block.content, markup=False, highlight=False)
        elif block.kind == "assistant":
            err_console.print(f"[bold cyan]{model_label}[/]")
            err_console.print(block.content, markup=False, highlight=False)
        elif block.kind == "reasoning_summary":
            elapsed = block.elapsed if block.elapsed is not None else 0.0
            if elapsed > 0.0:
                err_console.print(render_reasoning_line(elapsed, active=False))
            else:
                err_console.print("[dim]▸ Daydreamed[/dim]")
        elif block.kind == "reasoning_hint":
            err_console.print(f"[dim]{block.content}[/dim]")
        else:
            continue

        if index < len(visible_blocks) - 1:
            _print_gap("loose")

    if visible_blocks:
        _print_gap("loose")


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


def _read_live_boxed_message(*, cli_page_mode: str = "loose") -> str:
    buffer = ""
    multiline = False
    selected_command: str | None = None
    pending_escape = ""
    pending_escape_started_at: float | None = None

    with _raw_stdin():
        renderer = _make_menu_renderer(cli_page_mode)
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
                    cli_page_mode=cli_page_mode,
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
                            cli_page_mode=cli_page_mode,
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
                            cli_page_mode=cli_page_mode,
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
                            cli_page_mode=cli_page_mode,
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
                                cli_page_mode=cli_page_mode,
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
                                cli_page_mode=cli_page_mode,
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
                                cli_page_mode=cli_page_mode,
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
                                cli_page_mode=cli_page_mode,
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
                            cli_page_mode=cli_page_mode,
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
                            cli_page_mode=cli_page_mode,
                        )
                    )
        finally:
            renderer.finish()

    if buffer.strip():
        err_console.print(f"[bold green]You[/bold green]")
        err_console.print(buffer, markup=False, highlight=False)
        _print_gap(cli_page_mode)

    return buffer


def _select_effort(current: str, *, supported: bool, cli_page_mode: str = "loose") -> str:
    if not sys.stdin.isatty() or not err_console.is_terminal:
        return current

    options = list(EFFORT_LEVELS)
    index = options.index(current)
    pending_escape = ""
    pending_escape_started_at: float | None = None

    with _raw_stdin():
        renderer = _make_menu_renderer(cli_page_mode)
        try:
            renderer.render(
                build_effort_menu_lines(
                    current,
                    options[index],
                    supported=supported,
                    cli_page_mode=cli_page_mode,
                )
            )
            while True:
                pending_key, pending_escape, pending_escape_started_at = _drain_pending_escape(
                    pending_escape,
                    pending_escape_started_at,
                )
                if pending_key is None:
                    lines = build_effort_menu_lines(
                        current,
                        options[index],
                        supported=supported,
                        cli_page_mode=cli_page_mode,
                    )
                    if not _wait_for_menu_input(
                        renderer,
                        lines,
                        pending_escape=pending_escape,
                    ):
                        continue
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
                    _print_gap(cli_page_mode)
                    return current
                if _is_up_key(key):
                    index = (index - 1) % len(options)
                elif _is_down_key(key):
                    index = (index + 1) % len(options)
                elif key in ("A", "B", "C", "D", "[", "O"):
                    continue
                elif key in ("1", "2", "3", "4"):
                    index = int(key) - 1
                renderer.render(
                    build_effort_menu_lines(
                        current,
                        options[index],
                        supported=supported,
                        cli_page_mode=cli_page_mode,
                    )
                )
        finally:
            renderer.finish()

    _print_gap(cli_page_mode)
    return options[index]


def _select_cli_page_mode(current: str) -> str:
    if not sys.stdin.isatty() or not err_console.is_terminal:
        return current

    options = list(CLI_PAGE_LEVELS)
    index = options.index(current) if current in options else 0
    pending_escape = ""
    pending_escape_started_at: float | None = None

    with _raw_stdin():
        renderer = _make_menu_renderer(current)
        try:
            renderer.render(build_cli_page_menu_lines(current, options[index]))
            while True:
                pending_key, pending_escape, pending_escape_started_at = _drain_pending_escape(
                    pending_escape,
                    pending_escape_started_at,
                )
                if pending_key is None:
                    lines = build_cli_page_menu_lines(current, options[index])
                    if not _wait_for_menu_input(
                        renderer,
                        lines,
                        pending_escape=pending_escape,
                    ):
                        continue
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
                    _print_gap(current)
                    return current
                if _is_up_key(key):
                    index = (index - 1) % len(options)
                elif _is_down_key(key):
                    index = (index + 1) % len(options)
                elif key in ("A", "B", "C", "D", "[", "O"):
                    continue
                renderer.render(build_cli_page_menu_lines(current, options[index]))
        finally:
            renderer.finish()

    _print_gap(options[index])
    return options[index]


def _select_draft_mode(current: str, *, cli_page_mode: str = "loose") -> str:
    if not sys.stdin.isatty() or not err_console.is_terminal:
        return current

    options = list(_DRAFT_COMMAND_LEVELS)
    index = options.index(current) if current in options else 0
    pending_escape = ""
    pending_escape_started_at: float | None = None

    with _raw_stdin():
        renderer = _make_menu_renderer(cli_page_mode)
        try:
            renderer.render(build_draft_menu_lines(current, options[index]))
            while True:
                pending_key, pending_escape, pending_escape_started_at = _drain_pending_escape(
                    pending_escape,
                    pending_escape_started_at,
                )
                if pending_key is None:
                    lines = build_draft_menu_lines(current, options[index])
                    if not _wait_for_menu_input(
                        renderer,
                        lines,
                        pending_escape=pending_escape,
                    ):
                        continue
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
                    _print_gap(cli_page_mode)
                    return current
                if _is_up_key(key):
                    index = (index - 1) % len(options)
                elif _is_down_key(key):
                    index = (index + 1) % len(options)
                elif key in ("A", "B", "C", "D", "[", "O"):
                    continue
                renderer.render(build_draft_menu_lines(current, options[index]))
        finally:
            renderer.finish()

    _print_gap(cli_page_mode)
    return options[index]


class _InlineTerminalRenderer(BottomTerminalRenderer):
    def __init__(self, stream, *, collapse_on_finish: bool = False) -> None:
        super().__init__(
            stream,
            clear_on_finish=True,
            scroll_on_grow=True,
            scroll_on_shrink=True,
            scroll_on_first_render=True,
            collapse_on_finish=collapse_on_finish,
        )


def _make_menu_renderer(cli_page_mode: str):
    if _is_tight_cli_page_mode(cli_page_mode):
        return InlineFlowRenderer(sys.stderr, clear_on_finish=True)
    return _InlineTerminalRenderer(
        sys.stderr,
        collapse_on_finish=_is_tight_cli_page_mode(cli_page_mode),
    )


class _ReasoningParser:
    def __init__(self) -> None:
        self.reasoning_text = ""
        self.in_reasoning = False
        self.saw_reasoning = False
        self.saw_think_tags = False
        self._emitted_any_visible = False
        self._buffer = ""
        self._implicit_reasoning = False
        self._implicit_reasoning_chars = 0
        self._visible_tail = ""

    def _consume_reasoning(self, reasoning_parts: list[str]) -> bool | None:
        """Process buffer in reasoning mode.

        Returns: True=closed, None=need more data, False=consumed char.
        """
        if self._buffer.startswith(CLOSE_THINK_TAG):
            self._buffer = self._buffer[len(CLOSE_THINK_TAG):]
            self.in_reasoning = False
            self._implicit_reasoning = False
            return True
        if _starts_with_partial_tag(self._buffer, CLOSE_THINK_TAG):
            return None
        if self._implicit_reasoning and _should_end_implicit_reasoning(self.reasoning_text, self._buffer):
            self.in_reasoning = False
            self._implicit_reasoning = False
            return True
        char = self._buffer[0]
        reasoning_parts.append(char)
        self.reasoning_text += char
        self._buffer = self._buffer[1:]
        if self._implicit_reasoning:
            self._implicit_reasoning_chars += 1
            if self._implicit_reasoning_chars >= _IMPLICIT_REASONING_MAX_CHARS:
                self.in_reasoning = False
                self._implicit_reasoning = False
                return True
        return False

    def _try_explicit_tags(self, visible_parts: list[str], reasoning_parts: list[str], just_closed: bool) -> str | None:
        """Check for <think>/</ think> tags.

        Returns: "opened"/"closed"/"wait"/"skip"/None.
        """
        if self._buffer.startswith(OPEN_THINK_TAG):
            self.saw_reasoning = True
            self.saw_think_tags = True
            self.in_reasoning = True
            self._implicit_reasoning = False
            self._buffer = self._buffer[len(OPEN_THINK_TAG):]
            return "opened"

        close_index = self._buffer.find(CLOSE_THINK_TAG)
        open_index = self._buffer.find(OPEN_THINK_TAG)
        if (
            not self.saw_reasoning
            and close_index != -1
            and (open_index == -1 or close_index < open_index)
        ):
            prefix = self._buffer[:close_index]
            if prefix:
                reasoning_parts.append(prefix)
                self.reasoning_text += prefix
            self.saw_reasoning = True
            self.saw_think_tags = True
            self._buffer = self._buffer[close_index + len(CLOSE_THINK_TAG):]
            return "closed"

        if _starts_with_partial_tag(self._buffer, OPEN_THINK_TAG):
            return "wait"
        if self.saw_reasoning and self._buffer.startswith(CLOSE_THINK_TAG):
            self._buffer = self._buffer[len(CLOSE_THINK_TAG):]
            return "closed"
        if just_closed and self.saw_reasoning and self._buffer[:1] in ("\n", "\r"):
            self._buffer = self._buffer[1:]
            return "skip"
        if (
            not self.saw_reasoning
            and _starts_with_partial_tag(self._buffer, CLOSE_THINK_TAG)
        ):
            return "wait"
        if self.saw_reasoning and _starts_with_partial_tag(self._buffer, CLOSE_THINK_TAG):
            return "wait"
        return None

    def _try_implicit_reasoning(self) -> bool | None:
        """Check heuristics for implicit reasoning.

        Returns: True=entered, None=tentative(wait), False=no match.
        """
        if (
            not self.saw_reasoning
            and not self._emitted_any_visible
            and _looks_like_reasoning_prefix(self._buffer)
        ):
            self.saw_reasoning = True
            self.in_reasoning = True
            self._implicit_reasoning = True
            self._implicit_reasoning_chars = 0
            return True
        if (
            not self.saw_reasoning
            and not self._emitted_any_visible
            and _might_be_reasoning_prefix(self._buffer)
        ):
            return None
        if (
            _can_start_late_implicit_reasoning(self._visible_tail)
            and _looks_like_strong_reasoning_line(self._buffer)
        ):
            self.saw_reasoning = True
            self.in_reasoning = True
            self._implicit_reasoning = True
            self._implicit_reasoning_chars = 0
            return True
        if (
            _can_start_late_implicit_reasoning(self._visible_tail)
            and _might_be_strong_reasoning_line(self._buffer)
        ):
            return None
        return False

    def feed(self, chunk: str) -> tuple[str, str, bool]:
        self._buffer += chunk
        visible_parts: list[str] = []
        reasoning_parts: list[str] = []
        just_closed = False

        while self._buffer:
            if self.in_reasoning:
                result = self._consume_reasoning(reasoning_parts)
                if result is True:
                    just_closed = True
                    continue
                if result is None:
                    break
                continue

            tag = self._try_explicit_tags(visible_parts, reasoning_parts, just_closed)
            if tag == "opened":
                continue
            if tag == "closed":
                just_closed = True
                continue
            if tag == "wait":
                break
            if tag == "skip":
                continue

            implicit = self._try_implicit_reasoning()
            if implicit is True:
                continue
            if implicit is None:
                break

            char = self._buffer[0]
            visible_parts.append(char)
            self._buffer = self._buffer[1:]
            self._visible_tail = (self._visible_tail + char)[-8:]
            if not char.isspace():
                self._emitted_any_visible = True

        return "".join(visible_parts), "".join(reasoning_parts), just_closed

    def flush(self) -> tuple[str, str, bool]:
        """Emit any remaining buffered content at end-of-stream."""
        remaining = self._buffer
        self._buffer = ""
        if self.in_reasoning and self._implicit_reasoning:
            # Implicit reasoning never found exit → force close
            self.in_reasoning = False
            self._implicit_reasoning = False
            return remaining, "", True
        if self.in_reasoning:
            # Explicit <think> never closed → add remainder to reasoning
            if remaining:
                self.reasoning_text += remaining
            return "", remaining, False
        # Not in reasoning → emit as visible text
        return remaining, "", False


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


def _check_reasoning_prefix(text: str) -> str | None:
    """Check if text looks like a reasoning prefix.

    Returns: "definite" / "tentative" / None.
    """
    sample = text.lstrip()
    if not sample:
        return None
    lowered = sample.lower()
    # Definite checks (no length limit)
    if any(marker in lowered for marker in REASONING_PREFIX_CANDIDATES):
        return "definite"
    if _looks_like_meta_reasoning_sentence(sample):
        return "definite"
    if sample.startswith("1.") and "\n" in sample:
        return "definite"
    # Tentative checks (with length limit)
    if len(lowered) > 64:
        return None
    if any(candidate.startswith(lowered) for candidate in REASONING_PREFIX_CANDIDATES):
        return "tentative"
    if any(candidate.startswith(lowered) for candidate in _META_REASONING_PREFIX_CANDIDATES):
        return "tentative"
    return None


def _looks_like_reasoning_prefix(text: str) -> bool:
    return _check_reasoning_prefix(text) == "definite"


def _might_be_reasoning_prefix(text: str) -> bool:
    return _check_reasoning_prefix(text) == "tentative"


def _starts_with_partial_tag(text: str, tag: str) -> bool:
    if not text or len(text) >= len(tag):
        return False
    return tag.startswith(text)


def _looks_like_meta_reasoning_sentence(text: str) -> bool:
    sample = text.lstrip().lower()
    if not sample:
        return False
    if any(sample.startswith(candidate) for candidate in _META_REASONING_PREFIX_CANDIDATES):
        return True
    if sample.startswith("that's ") and (" line" in sample or " lines" in sample):
        return True
    if sample.startswith("okay, i will") and ("thinking block" in sample or "count" in sample):
        return True
    return False


def _should_end_implicit_reasoning(reasoning_text: str, buffer: str) -> bool:
    if not reasoning_text or not (reasoning_text.endswith("\n") or reasoning_text.endswith("\r")):
        return False
    if _last_reasoning_line_opens_draft(reasoning_text):
        return False
    sample = buffer.lstrip()
    if not sample:
        return False
    if sample.startswith(('"', "“", "'")):
        return False
    if _check_reasoning_line_start(sample) is not None:
        return False
    return True


def _last_reasoning_line_opens_draft(reasoning_text: str) -> bool:
    lines = reasoning_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for line in reversed(lines):
        sample = line.strip().lower()
        if not sample:
            continue
        return sample in (
            "response:",
            "response：",
            "final output:",
            "final output：",
            "final choice:",
            "final choice：",
            "final decision:",
            "final decision：",
            "answer:",
            "answer：",
            "回答:",
            "回答：",
            "答案:",
            "答案：",
            "回复:",
            "回复：",
        )
    return False


_REASONING_LINE_START_MARKERS = (
    "usually, this means",
    "this implies",
    "user says",
    "intent is",
    "response should",
    "keep it short",
    "thinking block:",
    "thinking:",
    "revised plan:",
    "final answer:",
    "response:",
    "answer:",
    "constraint check:",
    "thinking block lines:",
    "actually, looking at",
    "let's check the system instruction again",
    "let's check the instruction again",
    "the instruction says",
    "let's just respond",
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
    "回答：",
    "回答:",
    "答案：",
    "答案:",
    "回复：",
    "回复:",
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


def _check_reasoning_line_start(text: str) -> str | None:
    """Check if text looks like a reasoning line start.

    Returns: "definite" / "tentative" / None.
    """
    sample = text.lstrip()
    if not sample:
        return None
    lowered = sample.lower()
    # Definite checks (no length limit)
    if _looks_like_meta_reasoning_sentence(sample):
        return "definite"
    if sample.startswith(('"', "“", "‘")):
        return "definite"
    if sample.startswith(("**", "* ", "- ", "• ", "> ")):
        return "definite"
    if _looks_like_emphasized_reasoning_label(sample):
        return "definite"
    if re.match(r"^\d+[\.\)]\s", sample):
        return "definite"
    if any(lowered.startswith(marker) for marker in _REASONING_LINE_START_MARKERS):
        return "definite"
    # Tentative checks (with length limit)
    if len(lowered) > 48:
        return None
    if any(candidate.startswith(lowered) for candidate in _META_REASONING_PREFIX_CANDIDATES):
        return "tentative"
    if any(candidate.startswith(lowered) for candidate in EMPHASIZED_REASONING_PREFIX_CANDIDATES):
        return "tentative"
    if any(candidate.startswith(lowered) for candidate in _REASONING_LINE_START_MARKERS):
        return "tentative"
    if any(candidate.startswith(lowered) for candidate in ("* ", "- ", "• ", "> ")):
        return "tentative"
    if re.match(r"^\d+[\.\)]?$", lowered):
        return "tentative"
    return None


def _looks_like_reasoning_line_start(text: str) -> bool:
    return _check_reasoning_line_start(text) == "definite"


def _might_be_reasoning_line_start(text: str) -> bool:
    return _check_reasoning_line_start(text) == "tentative"


def _looks_like_emphasized_reasoning_label(text: str) -> bool:
    match = re.match(r"^\*([^*\n]{1,80})\*\s*", text)
    if not match:
        return False
    inner = match.group(1).strip().lower()
    if inner.endswith(":"):
        return True
    markers = (
        "simpler",
        "friendly",
        "let's",
        "final",
        "wait",
        "okay",
        "draft",
        "review",
        "analyze",
        "determine",
        "option",
        "self-correction",
    )
    return any(inner.startswith(marker) for marker in markers)


_STRONG_REASONING_MARKERS = (
    "usually, this means",
    "this implies",
    "user says",
    "intent is",
    "response should",
    "keep it short",
    "revised plan:",
    "thinking block:",
    "thinking:",
    "user:",
    "model:",
    "input:",
    "language:",
    "meaning:",
    "intent:",
    "final output:",
    "final answer:",
    "final choice:",
    "final decision:",
    "response:",
    "answer:",
    "constraint check:",
    "thinking block lines:",
    "actually, looking at",
    "let's check the system instruction again",
    "let's check the instruction again",
    "the instruction says",
    "let's just respond",
    "let me think",
    "my reasoning",
    "my reasoning:",
    "the question",
    "the question/task",
    "refinement:",
    "draft:",
    "review:",
    "analysis:",
    "reasoning:",
    "self-correction:",
    "wait,",
    "wait ",
    "okay,",
    "okay ",
    "输入：",
    "输入:",
    "最终输出：",
    "最终输出:",
    "最终决定：",
    "最终决定:",
    "回答：",
    "回答:",
    "答案：",
    "答案:",
    "回复：",
    "回复:",
    "总结：",
    "总结:",
    "用户问的是",
    "用户想",
    "用户的问题",
    "用户提出",
    "用户询问",
    "这个问题",
    "这道题",
    "让我",
    )


def _check_strong_reasoning_line(text: str) -> str | None:
    """Check if text looks like a strong reasoning line.

    Returns: "definite" / "tentative" / None.
    """
    sample = text.lstrip()
    if not sample:
        return None
    lowered = sample.lower()
    # Definite checks (no length limit)
    if _looks_like_meta_reasoning_sentence(sample):
        return "definite"
    if _looks_like_emphasized_reasoning_label(sample):
        return "definite"
    if any(lowered.startswith(marker) for marker in _STRONG_REASONING_MARKERS):
        return "definite"
    # Tentative checks (with length limit)
    if len(lowered) > 40:
        return None
    if any(candidate.startswith(lowered) for candidate in _META_REASONING_PREFIX_CANDIDATES):
        return "tentative"
    if any(candidate.startswith(lowered) for candidate in _STRONG_REASONING_MARKERS):
        return "tentative"
    return None


def _looks_like_strong_reasoning_line(text: str) -> bool:
    return _check_strong_reasoning_line(text) == "definite"


def _might_be_strong_reasoning_line(text: str) -> bool:
    return _check_strong_reasoning_line(text) == "tentative"


def _can_start_late_implicit_reasoning(visible_tail: str) -> bool:
    if not visible_tail:
        return False
    return visible_tail.endswith("\n") or visible_tail.endswith("\n\n")


def _extract_answer_labeled_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lowered = normalized.lower()
    best = ""
    best_index = -1
    for label in _FINAL_ANSWER_LABELS:
        index = lowered.rfind(label.lower())
        if index == -1:
            continue
        candidate = normalized[index + len(label):].strip()
        if candidate and index >= best_index:
            best = candidate
            best_index = index
    return best


def _paragraph_looks_like_reasoning(paragraph: str) -> bool:
    lines = [line.strip() for line in paragraph.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [line for line in lines if line]
    if not lines:
        return False

    reasoning_lines = 0
    for line in lines:
        lowered = line.lower()
        if _looks_like_strong_reasoning_line(line) or _looks_like_reasoning_line_start(line):
            reasoning_lines += 1
            continue
        if re.match(
            r"^\d+\.\s+(identify|recall|keep|ignore|determine|analyze|draft|check|review|summarize)\b",
            lowered,
        ):
            reasoning_lines += 1
            continue

    if reasoning_lines == 0:
        return False
    return reasoning_lines >= len(lines) or reasoning_lines >= max(2, len(lines) - 1)


def _clean_final_output_from_reasoning_leak(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    marker_indexes = [index for index, line in enumerate(lines) if _looks_like_strong_reasoning_line(line)]
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", normalized)
        if paragraph.strip()
    ]
    if not marker_indexes and not any(_paragraph_looks_like_reasoning(paragraph) for paragraph in paragraphs):
        return text
    if not paragraphs:
        return text
    for paragraph in reversed(paragraphs):
        labeled = _extract_answer_labeled_text(paragraph)
        if labeled:
            return labeled
        if not _paragraph_looks_like_reasoning(paragraph):
            return paragraph
    return text


def _recover_final_output(candidate_text: str, reasoning_text: str) -> str:
    visible = _clean_final_output_from_reasoning_leak(candidate_text).strip()
    if visible:
        return visible

    normalized = reasoning_text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", normalized)
        if paragraph.strip()
    ]
    for paragraph in reversed(paragraphs):
        if not _paragraph_looks_like_reasoning(paragraph):
            return paragraph
    return ""


def _should_animate_final_body() -> bool:
    if not isinstance(err_console, Console):
        return False
    stream = getattr(err_console, "file", None)
    return bool(
        err_console.is_terminal
        and stream is not None
        and getattr(stream, "isatty", lambda: False)()
    )


def _iter_body_chunks(text: str) -> list[str]:
    chunks: list[str] = []
    line_visible_count = 0
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        if char == "\n":
            chunks.append("\n")
            line_visible_count = 0
            index += 1
            continue

        if line_visible_count < 32:
            chunk_size = 1
        elif line_visible_count < 96:
            chunk_size = 2
        else:
            chunk_size = 3

        end = min(length, index + chunk_size)
        newline_index = text.find("\n", index, end)
        if newline_index != -1:
            end = newline_index
        if end == index:
            chunks.append("\n")
            line_visible_count = 0
            index += 1
            continue

        chunk = text[index:end]
        chunks.append(chunk)
        line_visible_count += len(chunk)
        index = end

    return chunks


def _print_final_body(text: str) -> None:
    if not text:
        return
    if not _should_animate_final_body():
        err_console.print(text, markup=False, highlight=False)
        return

    stream = err_console.file
    for chunk in _iter_body_chunks(text):
        stream.write(chunk)
        stream.flush()
        if chunk == "\n":
            time.sleep(0.018)
            continue

        delay = 0.008 if len(chunk) == 1 else 0.005
        if chunk[-1] in ",;，；:":
            delay += 0.006
        elif chunk[-1] in ".!?。！？":
            delay += 0.012
        time.sleep(delay)
    if not text.endswith("\n"):
        stream.write("\n")
        stream.flush()


def _finalize_output_text(full_text: str, candidate_text: str, parser: _ReasoningParser) -> str:
    """Determine final visible text after streaming.

    Three paths:
    1. Explicit tags -> trust separation, minimal cleanup
    2. Implicit reasoning -> _recover_final_output heuristic
    3. No reasoning -> _clean_final_output_from_reasoning_leak
    """
    if parser.saw_reasoning:
        if parser.saw_think_tags:
            result = full_text.strip()
            if not result:
                result = (candidate_text or "").strip()
            if not result:
                result = _recover_final_output("", parser.reasoning_text)
            if not result and parser.reasoning_text.strip():
                # Model put everything inside <think> with no visible answer.
                # Force-extract: try paragraphs first, then lines.
                paragraphs = [
                    p.strip() for p in re.split(r"\n\s*\n", parser.reasoning_text) if p.strip()
                ]
                if paragraphs:
                    result = paragraphs[-1]
                else:
                    # Single paragraph — use it directly
                    result = parser.reasoning_text.strip()
            return result
        return _recover_final_output(full_text or candidate_text, parser.reasoning_text)
    return _clean_final_output_from_reasoning_leak(full_text)


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
    draft_model=None,
    num_draft_tokens: int | None = None,
    prefill_step_size: int | None = None,
    prompt_cache=None,
    cli_page_mode: str = "loose",
):
    """Stream a response, collecting the full text."""
    full_text = ""
    candidate_text = ""
    last_response = None
    streamed_output = False
    stream_to_stdout = not err_console.is_terminal
    parser = _ReasoningParser()
    prev_reasoning = False
    # Check if the chat template already starts the model inside <think>.
    # Qwen3.5 templates with has_thinking=True append <think>\n to the prompt,
    # so the model's first generated tokens are reasoning content (no <think> tag).
    # Pre-initialize the parser in reasoning mode so these tokens are correctly
    # classified as reasoning instead of leaking to visible text.
    _prompt_starts_in_think = False
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            _probe = tokenizer.apply_chat_template(
                [{"role": "user", "content": "x"}],
                tokenize=False,
                add_generation_prompt=True,
                **(chat_template_kwargs or {}),
            )
            _prompt_starts_in_think = isinstance(_probe, str) and _probe.rstrip().endswith("<think>")
        except Exception:
            pass
    if _prompt_starts_in_think:
        parser.saw_reasoning = True
        parser.saw_think_tags = True
        parser.in_reasoning = True
    _had_reasoning = False
    _reasoning_elapsed = None

    with daydreaming_status(err_console, model_label, cli_page_mode=cli_page_mode, draft_active=draft_model is not None) as status:
        for response in engine.generate_stream(
            model, tokenizer, messages,
            max_tokens=max_tokens, temp=temp, top_p=top_p,
            chat_template_kwargs=chat_template_kwargs,
            draft_model=draft_model,
            num_draft_tokens=num_draft_tokens,
            prefill_step_size=prefill_step_size,
            prompt_cache=prompt_cache,
        ):
            raw_chunk = response.text or ""
            text_chunk, reasoning_chunk, reasoning_closed = parser.feed(raw_chunk)
            entered_reasoning = parser.in_reasoning and not prev_reasoning

            # Reasoning state transitions → drive the status display.
            has_real_reasoning = reasoning_chunk and reasoning_chunk.strip()
            first_reasoning = not status.had_reasoning
            if entered_reasoning:
                # On FIRST reasoning entry: clear echo/preamble that was
                # already streamed as visible text.
                # On re-entry: preserve already-streamed answer text.
                if first_reasoning and streamed_output and full_text.strip():
                    full_text = ""
                    if hasattr(status, "clear_output"):
                        status.clear_output()
                    elif hasattr(status, "output"):
                        try:
                            status.output = ""
                        except Exception:
                            pass
                    streamed_output = False
                candidate_text = ""
            # Start reasoning box (only when there's real non-whitespace content)
            if has_real_reasoning and not status.had_reasoning:
                status.start_reasoning()
            if reasoning_chunk:
                status.append_reasoning(reasoning_chunk)
            if reasoning_closed:
                # Retroactive reasoning: </think> found after visible text
                # was already streamed (model didn't output <think> first).
                # Move all streamed visible text into the reasoning box.
                if streamed_output and not status.had_reasoning and full_text.strip():
                    status.start_reasoning()
                    retroactive = full_text.strip()
                    status.append_reasoning(retroactive)
                    parser.reasoning_text = retroactive + "\n" + parser.reasoning_text
                    full_text = ""
                    candidate_text = ""
                    if hasattr(status, "clear_output"):
                        status.clear_output()
                    elif hasattr(status, "output"):
                        try:
                            status.output = ""
                        except Exception:
                            pass
                    streamed_output = False
                status.end_reasoning()
            prev_reasoning = parser.in_reasoning

            # Stats updates
            if response.prompt_tps and not streamed_output:
                status.update(phase="prefill", tokens_per_second=response.prompt_tps)

            # Stream visible text in real-time:
            # - Before any reasoning: stream directly
            # - After reasoning ended (</think> seen): stream directly
            # - During reasoning: buffer in candidate_text
            reasoning_ended = parser.saw_reasoning and not parser.in_reasoning
            if text_chunk and (not parser.saw_reasoning or reasoning_ended):
                if not streamed_output:
                    status.ensure_minimum_wait(0.28)
                    status.update(waiting=False, phase=None)
                full_text += text_chunk
                last_response = response
                if stream_to_stdout:
                    print(text_chunk, end="", flush=True)
                else:
                    status.append_output(text_chunk)
                streamed_output = True
                if response.generation_tps:
                    status.update(phase=None, tokens_per_second=response.generation_tps)
            elif text_chunk:
                # Inside reasoning or holding for <think> — buffer visible text
                candidate_text += text_chunk
                last_response = response
                if response.generation_tps:
                    status.update(phase=None, tokens_per_second=response.generation_tps)
            elif response.generation_tps:
                status.update(tokens_per_second=response.generation_tps)

            if response.finish_reason:
                break

        # Flush remaining parser buffer
        flush_visible, flush_reasoning, flush_closed = parser.flush()
        if flush_reasoning:
            status.append_reasoning(flush_reasoning)
        if flush_closed and not status.had_reasoning and full_text.strip():
            # Implicit reasoning consumed everything -> retroactive move
            status.start_reasoning()
            status.append_reasoning(full_text.strip())
            parser.reasoning_text = full_text.strip() + "\n" + parser.reasoning_text
            full_text = ""
            candidate_text = ""
            if hasattr(status, "clear_output"):
                status.clear_output()
            streamed_output = False
            status.end_reasoning()
        if flush_visible:
            full_text += flush_visible
            if not stream_to_stdout:
                status.append_output(flush_visible)
            streamed_output = True

        # Ensure reasoning display is properly finalized (handles cases
        # where <think> was opened but model stopped before </think>).
        # end_reasoning() is idempotent — safe to call even if already ended.
        if status.had_reasoning:
            status.end_reasoning()

        # Pre-compute final text and snapshot status WHILE overlay is still
        # active, so the teardown→print gap is minimal.
        full_text = _finalize_output_text(full_text, candidate_text, parser)
        full_text = full_text.lstrip("\n")
        if status.had_reasoning and full_text:
            full_text = full_text.rstrip()
        _had_reasoning = status.had_reasoning
        _reasoning_elapsed = status.reasoning_elapsed

    # Overlay is now torn down — print final output immediately.
    if not stream_to_stdout:
        if _had_reasoning and _reasoning_elapsed is not None:
            err_console.print(render_reasoning_line(_reasoning_elapsed, active=False))
            _print_gap(cli_page_mode)
        if full_text:
            err_console.print(f"[bold cyan]{model_label}[/bold cyan]")
            if streamed_output:
                # User already saw text stream live — print instantly,
                # no point re-animating what they already read.
                err_console.print(full_text, markup=False, highlight=False)
            else:
                # Text was extracted from reasoning (never shown live) —
                # use character animation for a polished reveal.
                _print_final_body(full_text)
            _print_gap(cli_page_mode)
        if _had_reasoning:
            err_console.print("[dim]Use /t to expand reasoning[/dim]")
            _print_gap(cli_page_mode)

    elif full_text and parser.saw_reasoning:
        print(full_text)

    if streamed_output and stream_to_stdout and not parser.saw_reasoning:
        print()

    if verbose and last_response:
        _print_stats(last_response)

    return full_text, last_response, parser.reasoning_text, _reasoning_elapsed


def _promote_terminal_output_to_scrollback() -> None:
    if not err_console.is_terminal:
        return
    stream = getattr(err_console, "file", sys.stderr)
    if not getattr(stream, "isatty", lambda: False)():
        return
    size = shutil.get_terminal_size(fallback=(96, 24))
    stream.write(f"\x1b[{size.lines};1H")
    stream.write("\n" * size.lines)
    stream.flush()


def _reserve_bottom_rows(count: int) -> None:
    if count <= 0 or not err_console.is_terminal:
        return
    stream = getattr(err_console, "file", sys.stderr)
    if not getattr(stream, "isatty", lambda: False)():
        return
    size = shutil.get_terminal_size(fallback=(96, 24))
    stream.write(f"\x1b[{size.lines};1H")
    stream.write("\n" * count)
    stream.flush()


def _prepare_status_overlay_space(model_label: str) -> None:
    _reserve_bottom_rows(_status_overlay_reserve_lines(model_label))


def _print_resumed_history(messages: list[ChatMessage], assistant_label: str, *, cli_page_mode: str = "loose") -> None:
    visible_messages = [message for message in messages if message.role in {"user", "assistant"}]
    if not visible_messages:
        return

    _print_gap(cli_page_mode)
    for message in visible_messages:
        if message.role == "user":
            err_console.print("[bold green]You[/]")
        else:
            err_console.print(f"[bold cyan]{assistant_label}[/]")
        err_console.print(message.content, markup=False, highlight=False)
        _print_gap(cli_page_mode)


def _select_dreaming_mode(*, cli_page_mode: str = "loose") -> str | None:
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
        renderer = _make_menu_renderer(cli_page_mode)
        try:
            renderer.render(build_dreaming_menu_lines(options[index], cli_page_mode=cli_page_mode))
            while True:
                pending_key, pending_escape, pending_escape_started_at = _drain_pending_escape(
                    pending_escape,
                    pending_escape_started_at,
                )
                if pending_key is None:
                    lines = build_dreaming_menu_lines(options[index], cli_page_mode=cli_page_mode)
                    if not _wait_for_menu_input(
                        renderer,
                        lines,
                        pending_escape=pending_escape,
                    ):
                        continue
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
                    _print_gap(cli_page_mode)
                    return None
                if _is_up_key(key):
                    index = (index - 1) % len(options)
                elif _is_down_key(key):
                    index = (index + 1) % len(options)
                elif key in ("A", "B", "C", "D", "[", "O"):
                    continue
                renderer.render(build_dreaming_menu_lines(options[index], cli_page_mode=cli_page_mode))
        finally:
            renderer.finish()

    _print_gap(cli_page_mode)
    return options[index]


def _select_session_action(
    sessions: list,
    *,
    allow_delete: bool = False,
    cli_page_mode: str = "loose",
) -> tuple[str, object] | None:
    """Arrow-key menu to choose a session action.

    Pattern: copy _select_effort() structure exactly.
    Returns (action, ChatSession) or None on Esc.
    """
    if not sys.stdin.isatty() or not err_console.is_terminal:
        return None

    if not sessions:
        return None

    index = 0
    pending_escape = ""
    pending_escape_started_at: float | None = None

    with _raw_stdin():
        renderer = _make_menu_renderer(cli_page_mode)
        try:
            window = _session_window(sessions, index)
            renderer.render(
                build_session_list_lines(
                    window,
                    index - _session_window_start(sessions, index),
                    allow_delete=allow_delete,
                    cli_page_mode=cli_page_mode,
                )
            )
            while True:
                pending_key, pending_escape, pending_escape_started_at = _drain_pending_escape(
                    pending_escape,
                    pending_escape_started_at,
                )
                if pending_key is None:
                    window = _session_window(sessions, index)
                    lines = build_session_list_lines(
                        window,
                        index - _session_window_start(sessions, index),
                        allow_delete=allow_delete,
                        cli_page_mode=cli_page_mode,
                    )
                    if not _wait_for_menu_input(
                        renderer,
                        lines,
                        pending_escape=pending_escape,
                    ):
                        continue
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
                    return ("resume", sessions[index])
                if key in ("\x03",):
                    raise KeyboardInterrupt
                if key == "\x1b":
                    renderer.finish()
                    _print_gap(cli_page_mode)
                    return None
                if _is_up_key(key):
                    index = (index - 1) % len(sessions)
                elif _is_down_key(key):
                    index = (index + 1) % len(sessions)
                elif allow_delete and key.lower() == "d":
                    return ("delete", sessions[index])
                elif key in ("A", "B", "C", "D", "[", "O"):
                    continue
                window = _session_window(sessions, index)
                renderer.render(
                    build_session_list_lines(
                        window,
                        index - _session_window_start(sessions, index),
                        allow_delete=allow_delete,
                        cli_page_mode=cli_page_mode,
                    )
                )
        finally:
            renderer.finish()

    _print_gap(cli_page_mode)
    return ("resume", sessions[index])


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
) -> list[Memory]:
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
    draft_mode: str | None = None,
    display_name: str | None = None,
) -> None:
    """Run a single generation and exit."""
    cli_page_mode = get_default_cli_page_mode()
    # Read from stdin if no prompt given
    if prompt is None:
        if sys.stdin.isatty():
            err_console.print("[red]Error:[/] No prompt provided.")
            raise SystemExit(1)
        prompt = sys.stdin.read().strip()
        if not prompt:
            err_console.print("[red]Error:[/] Empty input.")
            raise SystemExit(1)

    primary_was_local = is_model_available_locally(model_name)
    resolved_name = ensure_runtime_model(model_name, auto_pull=True, register_alias=True)
    just_downloaded_main = (not primary_was_local) and is_model_available_locally(resolved_name)
    with err_console.status("Preparing model..."):
        model, tokenizer = engine.load_model(resolved_name, ensure_available=False)
    if type(model).__module__.startswith("mlx_lm."):
        engine.set_metal_wired_limit()
    supports_draft_runtime, draft_runtime_reason = engine.speculative_runtime_status(model)
    if not supports_draft_runtime:
        draft_repo, num_draft_tokens = None, None
        if draft_mode == "force":
            err_console.print(f"[dim]{draft_runtime_reason}[/dim]")
    else:
        draft_repo, num_draft_tokens = _resolve_speculative_settings(
            model_name=model_name,
            resolved_model=resolved_name,
            draft_mode=draft_mode,
            just_downloaded_main=just_downloaded_main,
            allow_prompt=sys.stdin.isatty(),
        )
    with err_console.status("Preparing model..."):
        draft_model = (
            engine.load_model(draft_repo, ensure_available=False)[0]
            if draft_repo is not None
            else None
        )
    if draft_model is not None and type(model).__module__.startswith("mlx_lm."):
        with err_console.status("Warming up draft model..."):
            engine.warmup_draft_model(model, draft_model, tokenizer)

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
        draft_model=draft_model,
        num_draft_tokens=num_draft_tokens,
        prefill_step_size=2048,
        cli_page_mode=cli_page_mode,
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
    draft_mode: str | None = None,
    display_name: str | None = None,
) -> None:
    """Run an interactive chat REPL."""
    primary_was_local = is_model_available_locally(model_name)
    repo_id = ensure_runtime_model(model_name, auto_pull=True, register_alias=True)
    just_downloaded_main = (not primary_was_local) and is_model_available_locally(repo_id)
    with err_console.status("Preparing model..."):
        model, tokenizer = engine.load_model(repo_id, ensure_available=False)
    supports_draft_runtime, draft_runtime_reason = engine.speculative_runtime_status(model)
    if draft_mode == "force":
        if not supports_draft_runtime:
            err_console.print(f"[dim]{draft_runtime_reason}[/dim]")
            draft_repo, num_draft_tokens = None, None
        else:
            draft_repo, num_draft_tokens = _resolve_speculative_settings(
                model_name=model_name,
                resolved_model=repo_id,
                draft_mode=draft_mode,
                just_downloaded_main=just_downloaded_main,
                allow_prompt=True,
            )
    else:
        draft_repo, num_draft_tokens = None, None
    if type(model).__module__.startswith("mlx_lm."):
        engine.set_metal_wired_limit()
    with err_console.status("Preparing model..."):
        draft_model = (
            engine.load_model(draft_repo, ensure_available=False)[0]
            if draft_repo is not None
            else None
        )
    if draft_model is not None and type(model).__module__.startswith("mlx_lm."):
        with err_console.status("Warming up draft model..."):
            engine.warmup_draft_model(model, draft_model, tokenizer)
    short = display_name or reverse_lookup(repo_id) or model_name
    cli_page_mode = get_default_cli_page_mode()
    supported_draft_repo = draft_model_for_family(repo_id)
    active_num_draft_tokens = num_draft_tokens
    prompt_cache = engine.create_prompt_cache(model, draft_model)

    err_console.print(f"[bold cyan]{short}[/]")
    err_console.print("[dim]Use / for commands.[/dim]")
    _print_gap(cli_page_mode)

    messages: list[dict] = []
    transcript_blocks: list[_TranscriptBlock] = []
    last_reasoning = ""
    effort = initial_effort if initial_effort in EFFORT_LEVELS else "default"
    effort_supported = _model_supports_effort(repo_id)

    # Persistence state
    session: ChatSession | None = None  # None = memoryless mode (default)
    session_memories: list[Memory] = []

    while True:
        try:
            user_input = _read_boxed_message(cli_page_mode=cli_page_mode)
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
            transcript_blocks = []
            prompt_cache = engine.create_prompt_cache(model, draft_model)
            if session is not None:
                session.messages = []
                session.updated_at = time.time()
                save_session(session)
            if _is_tight_cli_page_mode(cli_page_mode):
                _repack_tight_page(short, transcript_blocks)
            else:
                _repack_loose_page(short, transcript_blocks)
            err_console.print("[dim]Chat history cleared.[/dim]")
            continue
        if stripped in ("/help", "/h", "/?"):
            err_console.print(f"[dim]{MULTILINE_SENTINEL}      — start/end multiline input[/dim]")
            err_console.print("[dim]\\        — continue input on the next line[/dim]")
            err_console.print("[dim]/new      — start a persistent chat session[/dim]")
            err_console.print("[dim]/resume   — resume a saved session[/dim]")
            err_console.print("[dim]/forget   — delete a saved session[/dim]")
            err_console.print("[dim]/rename   — rename a saved session[/dim]")
            err_console.print("[dim]/dreaming — consolidate memories from conversation[/dim]")
            err_console.print("[dim]/memory   — view extracted memories[/dim]")
            err_console.print("[dim]/effort   — pick instant / short / default / long[/dim]")
            err_console.print(f"[dim]{DRAFT_SLASH_COMMAND} — turn draft acceleration on / off[/dim]")
            err_console.print("[dim]/cli-page — choose loose / tight page spacing[/dim]")
            err_console.print("[dim]/reset    — clear conversation history[/dim]")
            err_console.print("[dim]/clear    — alias for /reset[/dim]")
            err_console.print("[dim]/t        — show last captured reasoning[/dim]")
            err_console.print("[dim]/quit     — exit chat[/dim]")
            err_console.print("[dim]/help     — show this help[/dim]")
            continue
        if stripped == "/":
            err_console.print(f"[dim]Type /effort, {DRAFT_SLASH_COMMAND}, /cli-page, /new, /resume, /rename, /dreaming, /memory, /t, /reset, or /help.[/dim]")
            continue
        if stripped.startswith("/effort"):
            parts = stripped.split(maxsplit=1)
            selected = _normalize_effort(parts[1]) if len(parts) > 1 else None
            if selected is None:
                selected = _select_effort(effort, supported=effort_supported, cli_page_mode=cli_page_mode)
            effort = selected
            prompt_cache = engine.create_prompt_cache(model, draft_model)
            note = "[dim]Model may ignore this setting.[/dim]" if not effort_supported else ""
            err_console.print(f"[dim]Reasoning effort set to {effort}.[/dim] {note}".rstrip())
            continue
        command_name = stripped.split(maxsplit=1)[0].lower()
        if command_name in {"/draft", DRAFT_SLASH_COMMAND}:
            parts = stripped.split(maxsplit=1)
            selected = _normalize_draft_command(parts[1]) if len(parts) > 1 else None
            if selected is None:
                current = "on" if draft_model is not None else "off"
                if len(parts) == 1:
                    selected = _select_draft_mode(current, cli_page_mode=cli_page_mode)
                else:
                    err_console.print(
                        f"[dim]Draft acceleration is {current}. Use {DRAFT_SLASH_COMMAND} on or {DRAFT_SLASH_COMMAND} off.[/dim]"
                    )
                    continue
            if supported_draft_repo is None:
                err_console.print("[dim]Draft acceleration is only supported for Qwen3.5 MLX models.[/dim]")
                continue
            if not supports_draft_runtime:
                reason = draft_runtime_reason or "Draft acceleration is unavailable for this runtime."
                err_console.print(f"[dim]{reason}[/dim]")
                continue
            if selected == "off":
                draft_model = None
                active_num_draft_tokens = None
                prompt_cache = engine.create_prompt_cache(model, None)
                err_console.print("[dim]Draft acceleration disabled.[/dim]")
                continue
            if not is_model_available_locally(supported_draft_repo):
                err_console.print(
                    f"[dim]Draft model not installed locally: {supported_draft_repo}. Pull it first to enable acceleration.[/dim]"
                )
                continue
            if draft_repo != supported_draft_repo or draft_model is None:
                with err_console.status("Loading draft model..."):
                    draft_model = engine.load_model(supported_draft_repo, ensure_available=False)[0]
                if type(model).__module__.startswith("mlx_lm."):
                    with err_console.status("Warming up draft model..."):
                        engine.warmup_draft_model(model, draft_model, tokenizer)
            draft_repo = supported_draft_repo
            active_num_draft_tokens = default_num_draft_tokens(repo_id)
            prompt_cache = engine.create_prompt_cache(model, draft_model)
            err_console.print(f"[dim]Draft acceleration enabled with {supported_draft_repo}.[/dim]")
            continue
        if stripped.startswith("/cli-page"):
            parts = stripped.split(maxsplit=1)
            selected = _normalize_cli_page_mode(parts[1]) if len(parts) > 1 else None
            if selected is None:
                selected = _select_cli_page_mode(cli_page_mode)
            cli_page_mode = set_default_cli_page_mode(selected)
            err_console.print(
                f"[dim]CLI page mode set to {cli_page_mode}. New chats will default to this mode.[/dim]"
            )
            transcript_blocks = _transcript_blocks_from_messages([
                ChatMessage(role=message["role"], content=message["content"], timestamp=0.0)
                for message in messages
                if message["role"] in {"user", "assistant"}
            ])
            if _is_tight_cli_page_mode(cli_page_mode):
                _repack_tight_page(short, transcript_blocks)
            else:
                _repack_loose_page(short, transcript_blocks)
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
            prompt_cache = engine.create_prompt_cache(model, draft_model)
            # Copy existing messages into session if any
            for msg in messages:
                session.messages.append(ChatMessage(
                    role=msg["role"], content=msg["content"], timestamp=time.time(),
                ))
            save_session(session)
            err_console.print(f"[dim]Persistent session started ({session.session_id[:8]}...)[/dim]")
            continue
        if stripped in ("/resume", "/forget"):
            while True:
                available_sessions = list_sessions()
                if not available_sessions:
                    err_console.print("[dim]No saved sessions.[/dim]")
                    break
                selection = _select_session_action(
                    available_sessions,
                    allow_delete=(stripped == "/forget"),
                    cli_page_mode=cli_page_mode,
                )
                if selection is None:
                    break
                action, selected_session = selection
                if stripped == "/forget":
                    action = "delete"
                if action == "delete":
                    if not _confirm_session_delete(selected_session):
                        continue
                    deleted = delete_session(selected_session.session_id)
                    if deleted:
                        if session is not None and session.session_id == selected_session.session_id:
                            session = None
                            session_memories = []
                            prompt_cache = engine.create_prompt_cache(model, draft_model)
                        err_console.print(
                            f"[dim]Deleted saved session: {selected_session.title or selected_session.session_id[:8]}[/dim]"
                        )
                    else:
                        err_console.print("[dim]Session was already removed.[/dim]")
                    continue
                session = selected_session
                messages = [{"role": m.role, "content": m.content} for m in session.messages]
                session_memories = load_memories(session.session_id)
                prompt_cache = engine.create_prompt_cache(model, draft_model)
                transcript_blocks = _transcript_blocks_from_messages(session.messages)
                err_console.print(f"[dim]Resumed: {session.title} ({len(messages)} messages)[/dim]")
                if _is_tight_cli_page_mode(cli_page_mode):
                    _repack_tight_page(short, transcript_blocks)
                else:
                    _repack_loose_page(short, transcript_blocks)
                break
            continue
        if stripped == "/rename":
            available_sessions = list_sessions()
            if not available_sessions:
                err_console.print("[dim]No saved sessions.[/dim]")
                continue
            selection = _select_session_action(
                available_sessions,
                allow_delete=False,
                cli_page_mode=cli_page_mode,
            )
            if selection is None:
                continue
            _, selected_session = selection
            old_title = selected_session.title or selected_session.session_id[:8]
            err_console.print(f"[dim]Current title: {old_title}[/dim]")
            try:
                new_title = err_console.input("[dim]New title: [/dim]").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if not new_title:
                err_console.print("[dim]Rename cancelled.[/dim]")
                continue
            renamed = rename_session(selected_session.session_id, new_title)
            if renamed:
                err_console.print(f"[dim]Renamed: {old_title} → {new_title}[/dim]")
                if session is not None and session.session_id == selected_session.session_id:
                    session.title = new_title
            else:
                err_console.print("[dim]Session not found.[/dim]")
            continue
        if stripped == "/dreaming":
            if session is None:
                err_console.print("[dim]Start a persistent session with /new before dreaming.[/dim]")
                continue
            if len(messages) < 2:
                err_console.print("[dim]Need at least one exchange to dream about.[/dim]")
                continue
            mode = _select_dreaming_mode(cli_page_mode=cli_page_mode)
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
                prompt_cache = engine.create_prompt_cache(model, draft_model)
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

        stream_result = _stream_response(
            model, tokenizer, request_messages,
            model_label=short,
            temp=temp, top_p=top_p, max_tokens=max_tokens, verbose=verbose,
            chat_template_kwargs=chat_template_kwargs,
            draft_model=draft_model,
            num_draft_tokens=active_num_draft_tokens,
            prefill_step_size=2048,
            prompt_cache=prompt_cache,
            cli_page_mode=cli_page_mode,
        )
        if len(stream_result) == 4:
            full_text, _, reasoning_text, reasoning_elapsed = stream_result
        else:
            full_text, _, reasoning_text = stream_result
            reasoning_elapsed = None
        last_reasoning = reasoning_text

        messages.append({"role": "assistant", "content": full_text})
        _append_transcript_turn(
            transcript_blocks,
            user_input=user_input,
            assistant_text=full_text,
            reasoning_elapsed=reasoning_elapsed,
            had_reasoning=bool(reasoning_text.strip()),
        )
        if _is_tight_cli_page_mode(cli_page_mode):
            _repack_tight_page(short, transcript_blocks)
        else:
            _repack_loose_page(short, transcript_blocks)

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

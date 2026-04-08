"""Interactive chat REPL and one-shot generation."""

from __future__ import annotations

import select
import sys
import re
import termios
import tty
from contextlib import contextmanager
from typing import Callable

from rich.console import Console

from daydream import engine
from daydream.models import ensure_runtime_model
from daydream.registry import reverse_lookup
from daydream.utils import (
    build_effort_menu_lines,
    build_input_box_lines,
    daydreaming_status,
    render_expanded_reasoning,
)

console = Console()
err_console = Console(stderr=True)

MULTILINE_SENTINEL = '"""'
OPEN_THINK_TAG = "<think>"
CLOSE_THINK_TAG = "</think>"
EFFORT_LEVELS = ("instant", "short", "default", "long")
SLASH_COMMANDS = (
    ("/effort", "adjust reasoning depth"),
    ("/help", "show available commands"),
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


def _effort_system_prompt(effort: str, model_name: str) -> str | None:
    if effort == "default" or not _model_supports_effort(model_name):
        return None

    prompts = {
        "instant": (
            "Reasoning effort: instant. Think only briefly, skip exploratory detours, "
            "and answer as soon as you have a solid result."
        ),
        "short": (
            "Reasoning effort: short. Use a compact reasoning chain, then answer directly."
        ),
        "long": (
            "Reasoning effort: long. Spend more time reasoning, check edge cases, and only "
            "then produce the final answer."
        ),
    }
    return prompts.get(effort)


def _build_request_messages(
    history: list[dict],
    *,
    system_prompt: str | None,
    effort: str,
    model_name: str,
) -> list[dict]:
    request_messages: list[dict] = []
    if system_prompt:
        request_messages.append({"role": "system", "content": system_prompt})
    effort_prompt = _effort_system_prompt(effort, model_name)
    if effort_prompt:
        request_messages.append({"role": "system", "content": effort_prompt})
    request_messages.extend(history)
    return request_messages


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
    first = sys.stdin.read(1)
    if first != "\x1b":
        return first

    fd = sys.stdin.fileno()
    if not select.select([fd], [], [], 0.02)[0]:
        return first
    second = sys.stdin.read(1)
    if second != "[":
        return first + second
    if not select.select([fd], [], [], 0.02)[0]:
        return first + second
    third = sys.stdin.read(1)
    return f"\x1b[{third}"


def _render_input_state(buffer: str, *, multiline: bool) -> object:
    normalized = buffer.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if normalized.endswith("\n"):
        lines.append("")
    command_rows = _matching_slash_commands(normalized) if not multiline else []
    return build_input_box_lines(
        lines,
        command_rows=command_rows,
        placeholder="Type a message or / for commands",
        multiline=multiline,
    )


def _read_live_boxed_message() -> str:
    buffer = ""
    multiline = False

    with _raw_stdin():
        renderer = _InlineTerminalRenderer(sys.stderr)
        try:
            renderer.render(_render_input_state(buffer, multiline=multiline))
            while True:
                key = _read_key()

                if key in ("\r", "\n"):
                    current_line = buffer.split("\n")[-1]
                    if multiline:
                        if current_line.strip() == MULTILINE_SENTINEL:
                            parts = buffer.split("\n")
                            buffer = "\n".join(parts[:-1]).strip()
                            break
                        buffer += "\n"
                        renderer.render(_render_input_state(buffer, multiline=multiline))
                        continue

                    if buffer.strip() == MULTILINE_SENTINEL:
                        buffer = ""
                        multiline = True
                        renderer.render(_render_input_state(buffer, multiline=multiline))
                        continue

                    if current_line.endswith("\\"):
                        buffer = buffer[:-1].rstrip() + "\n"
                        multiline = True
                        renderer.render(_render_input_state(buffer, multiline=multiline))
                        continue

                    buffer = buffer.strip()
                    break

                if key in ("\x7f", "\b"):
                    if buffer:
                        buffer = buffer[:-1]
                        renderer.render(_render_input_state(buffer, multiline=multiline))
                    continue

                if key == "\x03":
                    raise KeyboardInterrupt

                if key == "\x04":
                    if not buffer:
                        raise EOFError
                    continue

                if key.startswith("\x1b"):
                    continue

                if key == "\t":
                    buffer += "    "
                    renderer.render(_render_input_state(buffer, multiline=multiline))
                    continue

                if key.isprintable():
                    buffer += key
                    renderer.render(_render_input_state(buffer, multiline=multiline))
        finally:
            renderer.finish()

    err_console.print()
    return buffer


def _select_effort(current: str, *, supported: bool) -> str:
    if not sys.stdin.isatty() or not err_console.is_terminal:
        return current

    options = list(EFFORT_LEVELS)
    index = options.index(current)

    with _raw_stdin():
        renderer = _InlineTerminalRenderer(sys.stderr)
        try:
            renderer.render(build_effort_menu_lines(current, options[index], supported=supported))
            while True:
                key = _read_key()
                if key in ("\r", "\n"):
                    break
                if key in ("\x03",):
                    raise KeyboardInterrupt
                if key == "\x1b":
                    renderer.finish()
                    err_console.print()
                    return current
                if key in ("\x1b[A", "k"):
                    index = (index - 1) % len(options)
                elif key in ("\x1b[B", "j"):
                    index = (index + 1) % len(options)
                elif key in ("1", "2", "3", "4"):
                    index = int(key) - 1
                renderer.render(build_effort_menu_lines(current, options[index], supported=supported))
        finally:
            renderer.finish()

    err_console.print()
    return options[index]


class _InlineTerminalRenderer:
    def __init__(self, stream) -> None:
        self.stream = stream
        self._line_count = 0
        self._cursor_hidden = False

    def render(self, lines: list[str]) -> None:
        if not self._cursor_hidden:
            self.stream.write("\x1b[?25l")
            self._cursor_hidden = True
        if self._line_count:
            self.stream.write("\r")
            if self._line_count > 1:
                self.stream.write(f"\x1b[{self._line_count - 1}A")
            self.stream.write("\x1b[J")
        self.stream.write("\n".join(lines))
        self.stream.flush()
        self._line_count = len(lines)

    def finish(self) -> None:
        if self._line_count:
            self.stream.write("\r")
            if self._line_count > 1:
                self.stream.write(f"\x1b[{self._line_count - 1}A")
            self.stream.write("\x1b[J")
            self.stream.flush()
            self._line_count = 0
        if self._cursor_hidden:
            self.stream.write("\x1b[?25h")
            self.stream.flush()
            self._cursor_hidden = False


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


def _stream_response(model, tokenizer, messages, *, model_label, temp, top_p, max_tokens, verbose):
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


def run_oneshot(
    model_name: str,
    *,
    prompt: str | None = None,
    temp: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 4096,
    system: str | None = None,
    verbose: bool = False,
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

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    _stream_response(
        model, tokenizer, messages,
        model_label=reverse_lookup(resolved_name) or model_name,
        temp=temp, top_p=top_p, max_tokens=max_tokens, verbose=verbose,
    )


def run_chat(
    model_name: str,
    *,
    temp: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 4096,
    system: str | None = None,
    verbose: bool = False,
) -> None:
    """Run an interactive chat REPL."""
    repo_id = ensure_runtime_model(model_name, auto_pull=True, register_alias=True)
    with err_console.status("Preparing model..."):
        model, tokenizer = engine.load_model(repo_id, ensure_available=False)
    short = reverse_lookup(repo_id) or model_name

    err_console.print(f"[bold cyan]{short}[/]")
    err_console.print("[dim]Use / for commands.[/dim]")
    err_console.print()

    messages: list[dict] = []
    last_reasoning = ""
    effort = "default"
    effort_supported = _model_supports_effort(repo_id)

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
            err_console.print("[dim]Chat history cleared.[/dim]")
            continue
        if stripped in ("/help", "/h", "/?"):
            err_console.print(f"[dim]{MULTILINE_SENTINEL}      — start/end multiline input[/dim]")
            err_console.print("[dim]\\        — continue input on the next line[/dim]")
            err_console.print("[dim]/effort  — pick instant / short / default / long[/dim]")
            err_console.print("[dim]/reset  — clear conversation history[/dim]")
            err_console.print("[dim]/clear  — alias for /reset[/dim]")
            err_console.print("[dim]/t      — show last captured reasoning[/dim]")
            err_console.print("[dim]/quit   — exit chat[/dim]")
            err_console.print("[dim]/help   — show this help[/dim]")
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

        messages.append({"role": "user", "content": user_input})
        request_messages = _build_request_messages(
            messages,
            system_prompt=system,
            effort=effort,
            model_name=repo_id,
        )

        full_text, _, reasoning_text = _stream_response(
            model, tokenizer, request_messages,
            model_label=short,
            temp=temp, top_p=top_p, max_tokens=max_tokens, verbose=verbose,
        )
        last_reasoning = reasoning_text

        messages.append({"role": "assistant", "content": full_text})

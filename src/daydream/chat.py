"""Interactive chat REPL and one-shot generation."""

from __future__ import annotations

import sys
from typing import Callable

from rich.console import Console
from rich.rule import Rule
from rich.text import Text

from daydream import engine
from daydream.models import ensure_runtime_model
from daydream.registry import reverse_lookup
from daydream.utils import daydreaming_status

console = Console()
err_console = Console(stderr=True)

MULTILINE_SENTINEL = '"""'
OPEN_THINK_TAG = "<think>"
CLOSE_THINK_TAG = "</think>"


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


class _ReasoningParser:
    def __init__(self) -> None:
        self.raw_text = ""
        self.emitted_visible_length = 0
        self.in_reasoning = False
        self.saw_reasoning = False

    def feed(self, chunk: str) -> tuple[str, bool]:
        self.raw_text += chunk
        visible_text, in_reasoning = _extract_visible_text(self.raw_text)
        delta = visible_text[self.emitted_visible_length:]
        just_closed = self.in_reasoning and not in_reasoning
        self.emitted_visible_length = len(visible_text)
        self.in_reasoning = in_reasoning
        self.saw_reasoning = self.saw_reasoning or OPEN_THINK_TAG in self.raw_text
        return delta, just_closed


def _extract_visible_text(text: str) -> tuple[str, bool]:
    visible: list[str] = []
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

        if not in_reasoning:
            visible.append(text[i])
        i += 1

    return "".join(visible), in_reasoning


def _stream_response(model, tokenizer, messages, *, model_label, temp, top_p, max_tokens, verbose):
    """Stream a response, collecting the full text. Returns (full_text, last_response)."""
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
            text_chunk, reasoning_closed = parser.feed(raw_chunk)

            # Reasoning state transitions → drive the status display
            if parser.in_reasoning and not prev_reasoning:
                status.start_reasoning()
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

    return full_text, last_response


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
    err_console.print("[dim]>>> Send a message. Use /help for commands.[/dim]")
    err_console.print()

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})

    while True:
        try:
            user_input = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            err_console.print()
            break

        user_input = _collect_multiline_message(user_input)
        stripped = user_input.strip()
        if not stripped:
            continue

        # Commands
        if stripped in ("/quit", "/exit", "/q"):
            break
        if stripped in ("/reset", "/clear", "/r"):
            messages = messages[:1] if system else []
            err_console.print("[dim]Chat history cleared.[/dim]")
            continue
        if stripped in ("/help", "/h", "/?"):
            err_console.print(f"[dim]{MULTILINE_SENTINEL}      — start/end multiline input[/dim]")
            err_console.print("[dim]\\        — continue input on the next line[/dim]")
            err_console.print("[dim]/reset  — clear conversation history[/dim]")
            err_console.print("[dim]/clear  — alias for /reset[/dim]")
            err_console.print("[dim]/quit   — exit chat[/dim]")
            err_console.print("[dim]/help   — show this help[/dim]")
            continue

        messages.append({"role": "user", "content": user_input})
        err_console.print(Rule(Text(" Daydream ", style="bold cyan"), style="grey35"))

        full_text, _ = _stream_response(
            model, tokenizer, messages,
            model_label=short,
            temp=temp, top_p=top_p, max_tokens=max_tokens, verbose=verbose,
        )

        messages.append({"role": "assistant", "content": full_text})

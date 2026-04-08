"""Interactive chat REPL and one-shot generation."""

from __future__ import annotations

import sys
from typing import Callable

from rich.console import Console
from rich.text import Text

from daydream import engine
from daydream.models import ensure_runtime_model
from daydream.registry import reverse_lookup
from daydream.utils import daydreaming_status

console = Console()
err_console = Console(stderr=True)

MULTILINE_SENTINEL = '"""'


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


def _stream_response(model, tokenizer, messages, *, temp, top_p, max_tokens, verbose):
    """Stream a response, collecting the full text. Returns (full_text, last_response)."""
    full_text = ""
    last_response = None
    wrote_output = False
    with daydreaming_status(err_console):
        for response in engine.generate_stream(
            model, tokenizer, messages,
            max_tokens=max_tokens, temp=temp, top_p=top_p,
        ):
            text_chunk = response.text or ""
            if text_chunk and not wrote_output:
                err_console.print()
            full_text += text_chunk
            last_response = response

            if text_chunk:
                print(text_chunk, end="", flush=True)
                wrote_output = True

            if response.finish_reason:
                break

    if wrote_output:
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
        err_console.print(Text("Daydream", style="bold cyan"))

        full_text, _ = _stream_response(
            model, tokenizer, messages,
            temp=temp, top_p=top_p, max_tokens=max_tokens, verbose=verbose,
        )

        messages.append({"role": "assistant", "content": full_text})

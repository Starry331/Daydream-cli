"""Interactive terminal UI helpers using raw terminal input."""

from __future__ import annotations

import os
import sys
import tty
import termios


# ANSI helpers
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_GREEN = "\033[32m"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_SAVE_CURSOR = "\0337"
_RESTORE_CURSOR = "\0338"
_ERASE_BELOW = "\033[J"


def _read_key() -> str:
    """Read a single keypress, returning a string identifier."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "up"
            if seq == "[B":
                return "down"
            return "escape"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":
            return "ctrl-c"
        if ch == "\x7f":
            return "backspace"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _term_width() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def interactive_select(
    title: str,
    items: list[tuple[str, str]],
) -> tuple[str, str] | None:
    """Show an arrow-key navigable list. Returns (short_name, repo_id) or None."""
    if not items:
        return None

    selected = 0
    total = len(items)
    max_visible = min(total, 15)

    def _render() -> str:
        buf: list[str] = []
        buf.append(f"  {_BOLD}{title}{_RESET}")
        offset = 0
        if total > max_visible:
            offset = max(0, min(selected - max_visible // 2, total - max_visible))
        for i in range(offset, min(offset + max_visible, total)):
            short, repo_id = items[i]
            if i == selected:
                buf.append(f"  {_BOLD}{_CYAN}> {short}{_RESET}  {_DIM}{repo_id}{_RESET}")
            else:
                buf.append(f"    {short}  {_DIM}{repo_id}{_RESET}")
        if total > max_visible:
            buf.append(f"  {_DIM}({selected + 1}/{total}){_RESET}")
        buf.append(f"  {_DIM}↑/↓ move  Enter confirm  Esc cancel{_RESET}")
        return "\r\n".join(buf)

    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)

    # Print initial frame
    sys.stdout.write(_HIDE_CURSOR + _SAVE_CURSOR + _render())
    sys.stdout.flush()

    try:
        while True:
            key = _read_key()
            if key in ("ctrl-c", "escape"):
                sys.stdout.write(_RESTORE_CURSOR + _ERASE_BELOW + _SHOW_CURSOR)
                sys.stdout.flush()
                return None
            if key == "up":
                selected = (selected - 1) % total
            elif key == "down":
                selected = (selected + 1) % total
            elif key == "enter":
                sys.stdout.write(_RESTORE_CURSOR + _ERASE_BELOW + _SHOW_CURSOR)
                sys.stdout.flush()
                short, _repo = items[selected]
                sys.stdout.write(f"  {_GREEN}✓{_RESET} {_BOLD}{_CYAN}{short}{_RESET}\n")
                sys.stdout.flush()
                return items[selected]
            else:
                continue

            # Redraw in place
            sys.stdout.write(_RESTORE_CURSOR + _ERASE_BELOW + _render())
            sys.stdout.flush()
    except (EOFError, KeyboardInterrupt):
        sys.stdout.write(_RESTORE_CURSOR + _ERASE_BELOW + _SHOW_CURSOR)
        sys.stdout.flush()
        return None


def interactive_input(prompt: str) -> str | None:
    """Show a text input with inline editing. Returns the string or None on cancel."""
    clean = prompt.replace("[bold]", _BOLD).replace("[/bold]", _RESET).replace("[/]", _RESET)
    sys.stdout.write(f"  {clean} ")
    sys.stdout.flush()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    buf: list[str] = []
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == "\x03":  # Ctrl-C
                sys.stdout.write(f"\r\033[K{_SHOW_CURSOR}")
                sys.stdout.flush()
                return None
            if ch == "\x1b":  # Escape
                sys.stdin.read(2)
                sys.stdout.write(f"\r\033[K{_SHOW_CURSOR}")
                sys.stdout.flush()
                return None
            if ch in ("\r", "\n"):
                sys.stdout.write(f"\r\n{_SHOW_CURSOR}")
                sys.stdout.flush()
                result = "".join(buf).strip()
                return result if result else None
            if ch == "\x7f":  # Backspace
                if buf:
                    buf.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if ch.isprintable():
                buf.append(ch)
                sys.stdout.write(ch)
                sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

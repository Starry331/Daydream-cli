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
_ERASE_BELOW = "\033[J"


def _read_key() -> str:
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


def _get_cursor_row() -> int:
    """Query terminal for current cursor row using DSR."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        sys.stdout.write("\033[6n")
        sys.stdout.flush()
        resp = ""
        while True:
            ch = sys.stdin.read(1)
            resp += ch
            if ch == "R":
                break
        # Response is \033[row;colR
        parts = resp.lstrip("\033[").rstrip("R").split(";")
        return int(parts[0])
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _move_to_row(row: int) -> None:
    sys.stdout.write(f"\033[{row};1H")


def interactive_select(
    title: str,
    items: list[tuple[str, str]],
) -> tuple[str, str] | None:
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

    start_row = _get_cursor_row()
    sys.stdout.write(_HIDE_CURSOR + _render())
    sys.stdout.flush()

    try:
        while True:
            key = _read_key()
            if key in ("ctrl-c", "escape"):
                _move_to_row(start_row)
                sys.stdout.write(_ERASE_BELOW + _SHOW_CURSOR)
                sys.stdout.flush()
                return None
            if key == "up":
                selected = (selected - 1) % total
            elif key == "down":
                selected = (selected + 1) % total
            elif key == "enter":
                _move_to_row(start_row)
                sys.stdout.write(_ERASE_BELOW + _SHOW_CURSOR)
                sys.stdout.flush()
                short, repo_id = items[selected]
                sys.stdout.write(f"  {_GREEN}✓{_RESET} {_BOLD}{_CYAN}{short}{_RESET}\r\n")
                sys.stdout.flush()
                return items[selected]
            else:
                continue

            _move_to_row(start_row)
            sys.stdout.write(_ERASE_BELOW + _render())
            sys.stdout.flush()
    except (EOFError, KeyboardInterrupt):
        _move_to_row(start_row)
        sys.stdout.write(_ERASE_BELOW + _SHOW_CURSOR)
        sys.stdout.flush()
        return None


def interactive_menu(
    title: str,
    items: list[dict],
) -> dict | None:
    """Show a rich menu with title, description, and status per item.

    Each item dict has keys: name, description, status, key.
    Status is one of: "ready", "install", "not_installed".
    """
    if not items:
        return None

    selected = 0
    total = len(items)

    _STATUS_LABEL = {
        "ready": f"{_GREEN}●{_RESET}",
        "install": f"{_CYAN}↓{_RESET}",
        "not_installed": f"{_DIM}○{_RESET}",
    }

    def _render() -> str:
        buf: list[str] = []
        buf.append(f"  {_BOLD}{title}{_RESET}")
        buf.append("")
        for i, item in enumerate(items):
            dot = _STATUS_LABEL.get(item.get("status", "ready"), "")
            name = item["name"]
            desc = item.get("description", "")
            if i == selected:
                buf.append(f"  {_BOLD}{_CYAN}> {dot} {name}{_RESET}")
                if desc:
                    buf.append(f"      {_DIM}{desc}{_RESET}")
            else:
                buf.append(f"    {dot} {name}")
                if desc:
                    buf.append(f"      {_DIM}{desc}{_RESET}")
            buf.append("")
        buf.append(f"  {_DIM}↑/↓ move  Enter select  Esc cancel{_RESET}")
        return "\r\n".join(buf)

    start_row = _get_cursor_row()
    sys.stdout.write(_HIDE_CURSOR + _render())
    sys.stdout.flush()

    try:
        while True:
            key = _read_key()
            if key in ("ctrl-c", "escape"):
                _move_to_row(start_row)
                sys.stdout.write(_ERASE_BELOW + _SHOW_CURSOR)
                sys.stdout.flush()
                return None
            if key == "up":
                selected = (selected - 1) % total
            elif key == "down":
                selected = (selected + 1) % total
            elif key == "enter":
                _move_to_row(start_row)
                sys.stdout.write(_ERASE_BELOW + _SHOW_CURSOR)
                sys.stdout.flush()
                item = items[selected]
                sys.stdout.write(f"  {_GREEN}✓{_RESET} {_BOLD}{item['name']}{_RESET}\r\n")
                sys.stdout.flush()
                return item
            else:
                continue

            _move_to_row(start_row)
            sys.stdout.write(_ERASE_BELOW + _render())
            sys.stdout.flush()
    except (EOFError, KeyboardInterrupt):
        _move_to_row(start_row)
        sys.stdout.write(_ERASE_BELOW + _SHOW_CURSOR)
        sys.stdout.flush()
        return None


def interactive_input(prompt: str) -> str | None:
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
            if ch == "\x03":
                sys.stdout.write(f"\r\033[K{_SHOW_CURSOR}")
                sys.stdout.flush()
                return None
            if ch == "\x1b":
                sys.stdin.read(2)
                sys.stdout.write(f"\r\033[K{_SHOW_CURSOR}")
                sys.stdout.flush()
                return None
            if ch in ("\r", "\n"):
                sys.stdout.write(f"\r\n{_SHOW_CURSOR}")
                sys.stdout.flush()
                result = "".join(buf).strip()
                return result if result else None
            if ch == "\x7f":
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

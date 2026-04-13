"""Interactive terminal UI helpers using raw terminal input."""

from __future__ import annotations

import sys
import tty
import termios


_CYAN = "\033[36m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_GREEN = "\033[32m"


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


def _draw(lines: list[str], prev_count: int) -> int:
    """Clear previous output and draw new lines. Returns line count."""
    # Move cursor up to overwrite previous render
    if prev_count > 0:
        sys.stdout.write(f"\033[{prev_count}F")  # up to first line, col 1
    else:
        sys.stdout.write("\r")
    sys.stdout.write("\033[J")  # clear from cursor to end of screen

    output = "\r\n".join(lines)
    sys.stdout.write(output)
    sys.stdout.flush()
    return len(lines) - 1  # number of \r\n written


def interactive_select(
    title: str,
    items: list[tuple[str, str]],
) -> tuple[str, str] | None:
    if not items:
        return None

    selected = 0
    total = len(items)
    max_visible = min(total, 15)

    def _build() -> list[str]:
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
        return buf

    sys.stdout.write("\033[?25l")  # hide cursor
    lines = _build()
    prev = _draw(lines, 0)

    try:
        while True:
            key = _read_key()
            if key in ("ctrl-c", "escape"):
                _draw([], prev)
                sys.stdout.write("\033[?25h")
                sys.stdout.flush()
                return None
            if key == "up":
                selected = (selected - 1) % total
            elif key == "down":
                selected = (selected + 1) % total
            elif key == "enter":
                _draw([], prev)
                short, repo_id = items[selected]
                sys.stdout.write(f"  {_GREEN}✓{_RESET} {_BOLD}{_CYAN}{short}{_RESET}\033[?25h\r\n")
                sys.stdout.flush()
                return items[selected]
            else:
                continue
            lines = _build()
            prev = _draw(lines, prev)
    except (EOFError, KeyboardInterrupt):
        _draw([], prev)
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        return None


def interactive_menu(
    title: str,
    items: list[dict],
) -> dict | None:
    if not items:
        return None

    selected = 0
    total = len(items)

    _STATUS = {
        "ready": f"{_GREEN}●{_RESET}",
        "install": f"{_CYAN}↓{_RESET}",
        "not_installed": f"{_DIM}○{_RESET}",
    }

    def _build() -> list[str]:
        buf: list[str] = []
        buf.append(f"  {_BOLD}{title}{_RESET}")
        buf.append("")
        for i, item in enumerate(items):
            dot = _STATUS.get(item.get("status", "ready"), "")
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
        return buf

    sys.stdout.write("\033[?25l")
    lines = _build()
    prev = _draw(lines, 0)

    try:
        while True:
            key = _read_key()
            if key in ("ctrl-c", "escape"):
                _draw([], prev)
                sys.stdout.write("\033[?25h")
                sys.stdout.flush()
                return None
            if key == "up":
                selected = (selected - 1) % total
            elif key == "down":
                selected = (selected + 1) % total
            elif key == "enter":
                _draw([], prev)
                item = items[selected]
                sys.stdout.write(f"  {_GREEN}✓{_RESET} {_BOLD}{item['name']}{_RESET}\033[?25h\r\n")
                sys.stdout.flush()
                return item
            else:
                continue
            lines = _build()
            prev = _draw(lines, prev)
    except (EOFError, KeyboardInterrupt):
        _draw([], prev)
        sys.stdout.write("\033[?25h")
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
                sys.stdout.write(f"\r\033[K\033[?25h")
                sys.stdout.flush()
                return None
            if ch == "\x1b":
                sys.stdin.read(2)
                sys.stdout.write(f"\r\033[K\033[?25h")
                sys.stdout.flush()
                return None
            if ch in ("\r", "\n"):
                sys.stdout.write(f"\r\n\033[?25h")
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

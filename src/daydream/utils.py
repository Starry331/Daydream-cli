from __future__ import annotations

import random
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

SPINNER_FRAMES = ("◜", "◠", "◝", "◞", "◡", "◟")
DEFAULT_TERMINAL_TITLE = "Daydream"


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def format_time_ago(dt: datetime | float | int) -> str:
    """Format a datetime or POSIX timestamp as relative time."""
    now = datetime.now(timezone.utc)
    if isinstance(dt, (int, float)):
        dt = datetime.fromtimestamp(dt, tz=timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    diff = now - dt
    seconds = int(diff.total_seconds())
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hours ago"
    days = hours // 24
    if days < 30:
        return f"{days} days ago"
    months = days // 30
    return f"{months} months ago"


def is_interactive() -> bool:
    """Check if stdin is a TTY."""
    return sys.stdin.isatty()


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _mix_color(start: str, end: str, ratio: float) -> str:
    sr, sg, sb = _hex_to_rgb(start)
    er, eg, eb = _hex_to_rgb(end)
    r = int(sr + (er - sr) * ratio)
    g = int(sg + (eg - sg) * ratio)
    b = int(sb + (eb - sb) * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"


def render_daydreaming_text(frame: int = 0, *, rainbow: bool = False) -> Text:
    word = "Daydreaming"
    gradient = (
        ["#ff5ea0", "#ffa24c", "#ffe066", "#54d2a0", "#5bc0ff", "#9b7bff"]
        if rainbow
        else ["#4d78a8", "#64aee8", "#8fd8ff", "#ecf7ff"]
    )
    text = Text()

    pulse_curve = [0.28, 0.42, 0.62, 0.82, 1.0, 0.82, 0.62, 0.42]
    pulse = pulse_curve[frame % len(pulse_curve)]
    orb_color = _mix_color(gradient[0], gradient[-1], pulse)
    orb_char = "●" if pulse > 0.72 else "•" if pulse > 0.45 else "·"
    text.append(orb_char, style=f"bold {orb_color}")
    text.append(" ", style="dim")

    sweep_span = 4.0
    cycle_width = len(word) + 8
    center = (frame % cycle_width) - 3

    for idx, char in enumerate(word):
        distance = abs(idx - center)
        glow = max(0.0, 1.0 - (distance / sweep_span))
        palette_pos = idx / max(len(word) - 1, 1)
        color_index = palette_pos * (len(gradient) - 1)
        base = int(color_index)
        frac = color_index - base
        start = gradient[base]
        end = gradient[min(base + 1, len(gradient) - 1)]
        base_color = _mix_color(start, end, frac)
        highlight_color = _mix_color(base_color, "#ffffff", 0.85 if not rainbow else 0.35)
        final_color = _mix_color(base_color, highlight_color, glow)
        style_parts = ["bold" if glow > 0.22 else "not bold", final_color]
        if glow < 0.18:
            style_parts.append("dim")
        text.append(char, style=" ".join(style_parts))

    text.append(" ", style="dim")
    active_dot = frame % 3
    for idx in range(3):
        dot_glow = 1.0 if idx == active_dot else 0.35 if idx == (active_dot - 1) % 3 else 0.18
        dot_color = _mix_color(gradient[-2], "#ffffff", 0.4 * dot_glow)
        style = f"{'bold' if idx == active_dot else 'not bold'} {dot_color}"
        if idx != active_dot:
            style += " dim"
        text.append("•", style=style)

    return text


def render_title_text(label: str, frame: int = 0) -> str:
    spinner = SPINNER_FRAMES[frame % len(SPINNER_FRAMES)]
    return f"{spinner} {label}"


def _title_stream():
    if getattr(sys.stderr, "isatty", lambda: False)():
        return sys.stderr
    if getattr(sys.stdout, "isatty", lambda: False)():
        return sys.stdout
    return None


def set_terminal_title(title: str) -> None:
    stream = _title_stream()
    if stream is None:
        return
    stream.write(f"\033]0;{title}\007")
    stream.flush()


class TerminalTitleAnimator:
    def __init__(self, label: str):
        self.label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if _title_stream() is None:
            return
        set_terminal_title(render_title_text(self.label, 0))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        frame = 0
        while not self._stop.wait(0.12):
            frame += 1
            set_terminal_title(render_title_text(self.label, frame))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        set_terminal_title(DEFAULT_TERMINAL_TITLE)


class DaydreamingAnimator:
    def __init__(self, console: Console):
        self.console = console
        self._rainbow = random.random() < 0.3
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._live: Live | None = None

    def start(self) -> None:
        if not self.console.is_terminal:
            return
        self._live = Live(
            Group(render_daydreaming_text(0, rainbow=self._rainbow)),
            console=self.console,
            refresh_per_second=12,
            transient=True,
        )
        self._live.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        frame = 0
        while not self._stop.wait(0.08):
            frame += 1
            if self._live is not None:
                self._live.update(Group(render_daydreaming_text(frame, rainbow=self._rainbow)))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        if self._live is not None:
            self._live.stop()


@contextmanager
def daydreaming_status(console: Console):
    animator = DaydreamingAnimator(console)
    title_animator = TerminalTitleAnimator("Daydreaming")
    title_animator.start()
    animator.start()
    try:
        yield animator
    finally:
        animator.stop()
        title_animator.stop()


@contextmanager
def terminal_title_status(label: str):
    animator = TerminalTitleAnimator(label)
    animator.start()
    try:
        yield animator
    finally:
        animator.stop()

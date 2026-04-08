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
DAYDREAM_Z_FRAMES = (
    "Z",
    "Z  ZZ",
    "Z  ZZ  ZZZ",
    "  Z  ZZ  ZZZ",
    "Z   ZZ   ZZZ",
    "ZZ   ZZZ",
)
DEFAULT_TERMINAL_TITLE = "Daydream CLI"


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

    z_frame = DAYDREAM_Z_FRAMES[frame % len(DAYDREAM_Z_FRAMES)]
    for idx, char in enumerate(z_frame):
        if char == " ":
            text.append(char)
            continue
        palette_pos = idx / max(len(z_frame) - 1, 1)
        color_index = palette_pos * (len(gradient) - 1)
        base = int(color_index)
        frac = color_index - base
        start = gradient[base]
        end = gradient[min(base + 1, len(gradient) - 1)]
        color = _mix_color(start, end, frac)
        text.append(char, style=f"bold {color}")
    text.append("   ", style="dim")

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
        if glow < 0.26:
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


def render_title_text(label: str, frame: int = 0, *, frames: tuple[str, ...] | None = None) -> str:
    active_frames = frames or SPINNER_FRAMES
    prefix = active_frames[frame % len(active_frames)]
    return f"{prefix} Daydream CLI — {label}"


def render_status_footer(
    model_label: str,
    *,
    tokens_per_second: float | None = None,
    phase: str | None = None,
) -> Text:
    text = Text(style="dim")
    text.append("  ")
    text.append(model_label, style="dim")
    if phase:
        text.append("  ·  ", style="dim")
        text.append(phase, style="dim")
    if tokens_per_second:
        text.append("  ·  ", style="dim")
        text.append(f"{tokens_per_second:.1f} tok/s", style="dim")
    return text


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
    def __init__(self, label: str, *, frames: tuple[str, ...] | None = None):
        self.label = label
        self.frames = frames
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if _title_stream() is None:
            return
        set_terminal_title(render_title_text(self.label, 0, frames=self.frames))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        frame = 0
        while not self._stop.wait(0.12):
            frame += 1
            set_terminal_title(render_title_text(self.label, frame, frames=self.frames))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        set_terminal_title(DEFAULT_TERMINAL_TITLE)


class ConversationStatus:
    def __init__(self, console: Console, model_label: str):
        self.console = console
        self.model_label = model_label
        self._rainbow = random.random() < 0.3
        self._frame = 0
        self._phase: str | None = "thinking"
        self._tokens_per_second: float | None = None
        self._waiting = True
        self._output = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._live: Live | None = None
        self._lock = threading.Lock()
        self._title_animator = TerminalTitleAnimator("Daydreaming", frames=DAYDREAM_Z_FRAMES)

    def _render(self) -> Group:
        with self._lock:
            footer = render_status_footer(
                self.model_label,
                tokens_per_second=self._tokens_per_second,
                phase=self._phase,
            )
            if self._waiting:
                return Group(
                    render_daydreaming_text(self._frame, rainbow=self._rainbow),
                    footer,
                )
            if self._output:
                return Group(Text(self._output), footer)
            return Group(footer)

    def start(self) -> None:
        self._title_animator.start()
        if not self.console.is_terminal:
            return
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=12,
            transient=False,
        )
        self._live.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(0.08):
            with self._lock:
                self._frame += 1
            if self._live is not None:
                self._live.update(self._render())

    def update(self, *, phase: str | None = None, tokens_per_second: float | None = None, waiting: bool | None = None) -> None:
        with self._lock:
            if phase is not None:
                self._phase = phase
            if tokens_per_second is not None:
                self._tokens_per_second = tokens_per_second
            if waiting is not None and self._waiting != waiting:
                self._waiting = waiting
                if not waiting:
                    self._title_animator.stop()
        if self._live is not None:
            self._live.update(self._render())

    def append_output(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._output += text
        if self._live is not None:
            self._live.update(self._render())

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        if self._live is not None:
            self._live.update(self._render())
            self._live.stop()
        self._title_animator.stop()


@contextmanager
def daydreaming_status(console: Console, model_label: str):
    animator = ConversationStatus(console, model_label)
    animator.start()
    try:
        yield animator
    finally:
        animator.stop()


@contextmanager
def terminal_title_status(label: str):
    animator = TerminalTitleAnimator(label)
    animator.start()
    try:
        yield animator
    finally:
        animator.stop()

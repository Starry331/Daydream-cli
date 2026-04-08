from __future__ import annotations

import math
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from rich.console import Console, Group
from rich.live import Live
from rich.rule import Rule
from rich.text import Text

# ── Constants ──────────────────────────────────────────────────────────

SPINNER_FRAMES = ("◜", "◠", "◝", "◞", "◡", "◟")
DEFAULT_TERMINAL_TITLE = "Daydream CLI"

# Z animation: single evolving cluster Z → ZZ → ZZZ → ZZ → Z
_Z_CYCLE = 32

# Dream color palette (dark → glow)
_C_DARK = "#1e3a5f"
_C_DIM = "#3d6490"
_C_MID = "#5a9bd5"
_C_BRIGHT = "#7ec8e3"
_C_GLOW = "#b8e6ff"
_C_WHITE = "#e0f2ff"
_C_REASON = "#6b7b8d"

# Title Z frames (single cluster)
_TITLE_Z_FRAMES = ("z", "zz", "zzz", "zz", "z", "zz", "zzz", "zz")

# Sentinel for "not provided" in update()
_UNSET = object()


# ── Generic utilities ──────────────────────────────────────────────────

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


# ── Color helpers ──────────────────────────────────────────────────────

def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _mix_color(start: str, end: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    sr, sg, sb = _hex_to_rgb(start)
    er, eg, eb = _hex_to_rgb(end)
    r = int(sr + (er - sr) * ratio)
    g = int(sg + (eg - sg) * ratio)
    b = int(sb + (eb - sb) * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"


def _smoothstep(t: float) -> float:
    """Hermite smoothstep for buttery animations."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _z_length_value(frame: int) -> float:
    """Animated visible cluster length in [1, 3]."""
    phase = (frame % _Z_CYCLE) / _Z_CYCLE
    if phase < 0.5:
        t = _smoothstep(phase / 0.5)
        return 1.0 + 2.0 * t
    t = _smoothstep((phase - 0.5) / 0.5)
    return 3.0 - 2.0 * t


def _z_slot_brightness(frame: int, slot: int) -> float:
    """Brightness [0, 1] for a single Z slot in the evolving cluster."""
    length_value = _z_length_value(frame)
    activation = max(0.0, min(1.0, length_value - slot))
    if activation <= 0.0:
        return 0.0

    breathe_wave = 0.5 + 0.5 * math.sin((frame / 16.0 + slot * 0.11) * math.tau)
    breathe = 0.08 * _smoothstep(breathe_wave)
    shimmer_wave = 0.5 + 0.5 * math.sin((frame / 5.4) + slot * 1.15)
    shimmer = 0.06 * (_smoothstep(shimmer_wave) - 0.5)
    brightness = 0.62 + 0.28 * activation + breathe + shimmer
    return max(0.0, min(1.0, brightness))


def _brightness_to_color(b: float) -> str:
    """Map brightness 0-1 → dream palette color."""
    if b <= 0.0:
        return _C_DARK
    if b <= 0.25:
        return _mix_color(_C_DARK, _C_DIM, b / 0.25)
    if b <= 0.5:
        return _mix_color(_C_DIM, _C_MID, (b - 0.25) / 0.25)
    if b <= 0.75:
        return _mix_color(_C_MID, _C_BRIGHT, (b - 0.5) / 0.25)
    return _mix_color(_C_BRIGHT, _C_GLOW, (b - 0.75) / 0.25)


# ── Render functions ───────────────────────────────────────────────────

def render_daydreaming_text(frame: int = 0) -> Text:
    """Render a dreamy Z cluster, a sweeping Daydreaming glow, and pulsing dots."""
    text = Text()
    text.append("  ")

    def dreamy_color(brightness: float) -> str:
        level = max(0.0, min(1.0, brightness))
        color = _brightness_to_color(level)
        if level < 0.18:
            color = _mix_color("#08111a", color, _smoothstep(level / 0.18))
        if level > 0.82:
            color = _mix_color(color, _C_WHITE, _smoothstep((level - 0.82) / 0.18))
        return color

    def dreamy_style(brightness: float) -> str:
        color = dreamy_color(brightness)
        r, g, b = _hex_to_rgb(color)
        luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        if brightness > 0.84 or luminance > 0.78:
            return f"bold {color}"
        return color

    # ── single Z cluster ──
    for slot in range(3):
        char_brightness = _z_slot_brightness(frame, slot)
        if char_brightness > 0.01:
            text.append("Z", style=dreamy_style(char_brightness))
        else:
            text.append(" ")

    text.append("  ")

    # ── "Daydreaming" with left-to-right sweep glow ──
    word = "Daydreaming"
    sweep_cycle = 22.0
    trail_width = 4.4
    core_width = 1.6
    travel = len(word) + trail_width * 2.0
    center = ((frame % sweep_cycle) / sweep_cycle) * travel - trail_width

    for idx, char in enumerate(word):
        dist = abs(idx - center)
        trail = _smoothstep(max(0.0, 1.0 - dist / trail_width))
        core = _smoothstep(max(0.0, 1.0 - dist / core_width))
        ambient_wave = 0.5 + 0.5 * math.sin((frame / 18.0 + idx * 0.08) * math.tau)
        ambient = 0.03 * _smoothstep(ambient_wave)
        brightness = min(1.0, 0.18 + ambient + 0.28 * trail + 0.46 * core)
        color = dreamy_color(brightness)
        if core > 0.0:
            color = _mix_color(color, _C_WHITE, 0.35 * core)
        r, g, b = _hex_to_rgb(color)
        luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        style = f"bold {color}" if core > 0.75 or luminance > 0.8 else color
        text.append(char, style=style)

    # ── Pulsing dots ··· ──
    text.append(" ")
    for i in range(3):
        wave = 0.5 + 0.5 * math.sin((frame / 18.0 + i * 0.18) * math.tau)
        pulse = _smoothstep(wave)
        dot_brightness = 0.22 + 0.30 * pulse
        text.append("·", style=dreamy_style(dot_brightness))

    return text


def render_reasoning_line(elapsed: float, *, active: bool = True) -> Text:
    """Collapsed reasoning indicator.

    active=True  →  ▸ daydreaming ··· (2.3s)
    active=False →  ▸ Daydreamed for 3.2s
    """
    text = Text()
    text.append("  ")

    if active:
        text.append("▸ ", style=f"bold {_C_REASON}")
        text.append("daydreaming ", style=_C_REASON)
        n_dots = int(elapsed * 2) % 4
        text.append("·" * n_dots, style=_C_REASON)
        text.append(" " * (3 - n_dots))
        text.append(f" ({elapsed:.1f}s)", style=f"dim {_C_REASON}")
    else:
        text.append("▸ ", style=f"dim {_C_REASON}")
        text.append(f"Daydreamed for {elapsed:.1f}s", style=f"dim {_C_REASON}")

    return text


def render_frame_rule(title=None) -> Rule:
    return Rule(title, style="grey27")


def render_reasoning_box(reasoning_text: str, frame: int) -> Group:
    """Render a framed reasoning box with animated header and clipped body."""
    normalized = reasoning_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if normalized.endswith("\n"):
        lines.append("")
    visible_lines = lines[-5:] if lines else [""]
    if not any(line.strip() for line in visible_lines):
        visible_lines = [""] * 4 + [" "]
    while len(visible_lines) < 5:
        visible_lines.insert(0, "")

    body = Text("\n".join(visible_lines), style=f"dim {_C_REASON}")
    return Group(
        render_frame_rule(render_daydreaming_text(frame)),
        body,
        render_frame_rule(),
    )


def render_input_box_header() -> Rule:
    return render_frame_rule(Text(" Chat ", style="bold grey58"))


def render_input_box_footer() -> Rule:
    return render_frame_rule(Text(" Send ", style="grey42"))


def render_input_box(
    lines: list[str],
    *,
    command_rows: list[tuple[str, str]] | None = None,
    placeholder: str = "Type a message",
    multiline: bool = False,
) -> Group:
    parts: list = [render_input_box_header()]

    if not lines:
        lines = [""]

    for index, line in enumerate(lines):
        prefix = "│ "
        if not line and index == 0:
            parts.append(Text.assemble((prefix, "grey50"), (placeholder, "dim")))
        else:
            parts.append(Text.assemble((prefix, "grey50"), (line or " ", "default")))

    if command_rows:
        parts.append(Text())
        parts.append(Text("  Commands", style="dim"))
        for name, description in command_rows:
            row = Text("  ", style="dim")
            row.append(name, style="grey70")
            row.append("  ")
            row.append(description, style="dim")
            parts.append(row)
    elif multiline:
        parts.append(Text())
        parts.append(Text("  Multiline mode · End with \"\"\"", style="dim"))

    parts.append(render_input_box_footer())
    return Group(*parts)


def render_effort_menu(
    current: str,
    selected: str,
    *,
    supported: bool,
) -> Group:
    parts: list = [render_frame_rule(Text(" Reasoning Effort ", style="bold grey58"))]
    options = ("instant", "short", "default", "long")
    for option in options:
        marker = "› " if option == selected else "  "
        style = "bold grey84" if option == selected else "dim"
        label = f"{marker}{option}"
        if option == current:
            label += "  current"
        parts.append(Text(label, style=style))

    parts.append(Text())
    parts.append(Text("  Use ↑/↓ or j/k · Enter to apply · Esc to cancel", style="dim"))
    if not supported:
        parts.append(Text("  This model may ignore effort controls.", style="dim"))
    parts.append(render_frame_rule())
    return Group(*parts)


def render_expanded_reasoning(reasoning_text: str) -> Group:
    normalized = reasoning_text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    body = Text(normalized or "(no reasoning captured)", style=f"dim {_C_REASON}")
    return Group(
        render_frame_rule(Text(" Reasoning ", style=f"bold {_C_REASON}")),
        body,
        render_frame_rule(),
    )


def render_status_footer(
    model_label: str,
    *,
    tokens_per_second: float | None = None,
    phase: str | None = None,
    hint: str | None = None,
) -> Text:
    """Status footer pinned at bottom of display area."""
    text = Text(style="dim")
    text.append("  ")
    text.append(model_label, style="dim")
    if phase:
        text.append("  ·  ", style="dim")
        text.append(phase, style="dim")
    if tokens_per_second:
        text.append("  ·  ", style="dim")
        text.append(f"{tokens_per_second:.1f} tok/s", style="dim")
    if hint:
        text.append("  ·  ", style="dim")
        text.append(hint, style="dim")
    return text


def render_title_text(label: str, frame: int = 0, *, frames: tuple[str, ...] | None = None) -> str:
    active_frames = frames or SPINNER_FRAMES
    prefix = active_frames[frame % len(active_frames)]
    return f"{prefix} Daydream CLI — {label}"


# ── Terminal title ─────────────────────────────────────────────────────

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
        self.frames = frames or SPINNER_FRAMES
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


# ── Conversation status display ────────────────────────────────────────
#
# Layout during reasoning:
#     ZZZ  Daydreaming ···
#     ┌──────────────────────────────┐
#     │ streamed reasoning...        │
#     │ newest line at bottom        │
#     └──────────────────────────────┘
#
#     model · 45.2 tok/s
#
# Layout during output (after reasoning):
#     ▸ Daydreamed for 3.2s
#
#     Streaming response text...
#
#     model · 45.2 tok/s
#
# Layout during output (no reasoning):
#     Response text...
#
#     model · 45.2 tok/s
#

class ConversationStatus:
    def __init__(self, console: Console, model_label: str):
        self.console = console
        self.model_label = model_label
        self._frame = 0
        self._phase: str | None = "thinking"
        self._tokens_per_second: float | None = None
        self._waiting = True
        self._output = ""
        self._reasoning_text = ""

        # Reasoning
        self._reasoning_active = False
        self._reasoning_start: float | None = None
        self._reasoning_elapsed: float | None = None
        self._had_reasoning = False

        # Threading
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._live: Live | None = None
        self._lock = threading.Lock()
        self._title_animator = TerminalTitleAnimator("Daydreaming", frames=_TITLE_Z_FRAMES)

    def _reasoning_time(self) -> float:
        elapsed = self._reasoning_elapsed or 0.0
        if self._reasoning_start is not None:
            return elapsed + (time.monotonic() - self._reasoning_start)
        return elapsed

    def _render(self) -> Group:
        with self._lock:
            parts: list = []

            if self._reasoning_active:
                parts.append(render_reasoning_box(self._reasoning_text, self._frame))
                parts.append(Text())
            elif self._waiting and not self._had_reasoning:
                parts.append(render_reasoning_box("", self._frame))
                parts.append(Text())
            else:
                # ── Reasoning summary (if model used thinking) ──
                if self._had_reasoning and self._reasoning_elapsed is not None:
                    parts.append(render_reasoning_line(self._reasoning_elapsed, active=False))
                    parts.append(Text())

                # ── Streamed output ──
                if self._output:
                    parts.append(Text(self._output))
                    parts.append(Text())

            # ── Footer (always at bottom) ──
            reasoning_hint = None
            if self._reasoning_active or self._had_reasoning:
                reasoning_hint = "Use /t to expand reasoning"
            parts.append(render_status_footer(
                self.model_label,
                tokens_per_second=self._tokens_per_second,
                hint=reasoning_hint,
            ))

            return Group(*parts)

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

    def update(
        self,
        *,
        phase=_UNSET,
        tokens_per_second=_UNSET,
        waiting: bool | None = None,
    ) -> None:
        with self._lock:
            if phase is not _UNSET:
                self._phase = phase
            if tokens_per_second is not _UNSET:
                self._tokens_per_second = tokens_per_second
            if waiting is not None and self._waiting != waiting:
                self._waiting = waiting
                if not waiting:
                    self._title_animator.stop()
        if self._live is not None:
            self._live.update(self._render())

    def start_reasoning(self) -> None:
        with self._lock:
            self._reasoning_active = True
            if self._reasoning_start is None:
                self._reasoning_start = time.monotonic()
            self._had_reasoning = True
            self._phase = "thinking"
        if self._live is not None:
            self._live.update(self._render())

    def end_reasoning(self) -> None:
        with self._lock:
            self._reasoning_active = False
            if self._reasoning_start is not None:
                elapsed = time.monotonic() - self._reasoning_start
                self._reasoning_elapsed = (self._reasoning_elapsed or 0.0) + elapsed
                self._reasoning_start = None
        self._title_animator.stop()
        if self._live is not None:
            self._live.update(self._render())

    def append_reasoning(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._reasoning_text += text
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
    status = ConversationStatus(console, model_label)
    status.start()
    try:
        yield status
    finally:
        status.stop()


@contextmanager
def terminal_title_status(label: str):
    animator = TerminalTitleAnimator(label)
    animator.start()
    try:
        yield animator
    finally:
        animator.stop()

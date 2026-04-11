from __future__ import annotations

import colorsys
import math
import random
import re
import select
import shutil
import sys
import threading
import time
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterable

from rich.console import Console, Group
from rich.rule import Rule
from rich.text import Text

# ── Constants ──────────────────────────────────────────────────────────

SPINNER_FRAMES = ("◜", "◠", "◝", "◞", "◡", "◟")
DEFAULT_TERMINAL_TITLE = "Daydream CLI"

# Z animation: single evolving cluster Z → ZZ → ZZZ → blank
_Z_CYCLE = 48

# Dream color palette (dark → glow)
_C_DARK = "#1e3a5f"
_C_DIM = "#3d6490"
_C_MID = "#5a9bd5"
_C_BRIGHT = "#7ec8e3"
_C_GLOW = "#b8e6ff"
_C_WHITE = "#e0f2ff"
_C_REASON = "#6b7b8d"
_RAINBOW_PROBABILITY = 0.33
_DREAM_WORDS = ("Daydreaming", "Imagining", "Floating", "Wandering", "Rêvant")

# Sleep phase palettes
_C_N3_DARK = "#0d1b2a"    # Deep navy — deep sleep
_C_N3_MID = "#1b3a5c"
_C_N3_GLOW = "#2e5f8a"
_C_N2_DARK = "#2d1b4e"    # Purple — spindle processing
_C_N2_MID = "#5c3d8f"
_C_N2_GLOW = "#8b5cf6"
_C_REM_DARK = "#3b1029"   # Crimson/warm — REM integration
_C_REM_MID = "#7c2d5c"
_C_REM_GLOW = "#d4507a"

SLEEP_PHASE_PALETTES = {
    "reming": (_C_MID, _C_BRIGHT, _C_GLOW),
    "n3":     (_C_N3_DARK, _C_N3_MID, _C_N3_GLOW),
    "n2":     (_C_N2_DARK, _C_N2_MID, _C_N2_GLOW),
    "rem":    (_C_REM_DARK, _C_REM_MID, _C_REM_GLOW),
}

SLEEP_PHASE_LABELS = {
    "reming": "REMing",
    "n3": "N3 \u00b7 Deep Sleep",
    "n2": "N2 \u00b7 Core Sleep",
    "rem": "REM \u00b7 Integration",
}
REFLECTING_LABEL = "Reflecting"

# Title Z frames (single cluster)
_TITLE_Z_FRAMES = ("z", "zz", "zzz", "zz", "z", "zz", "zzz", "zz")

# Sentinel for "not provided" in update()
_UNSET = object()
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"
_FRAME_CHAR = "_"
CLI_PAGE_MODES = ("loose", "tight")


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


def _rainbow_color(position: float, brightness: float) -> str:
    hue = position % 1.0
    saturation = 0.62
    value = 0.24 + 0.56 * max(0.0, min(1.0, brightness))
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _smoothstep(t: float) -> float:
    """Hermite smoothstep for buttery animations."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _display_width(text: str) -> int:
    width = 0
    for char in _strip_ansi(text):
        if unicodedata.combining(char):
            continue
        if unicodedata.east_asian_width(char) in ("W", "F"):
            width += 2
        else:
            width += 1
    return width


def _fit_display_width(text: str, width: int) -> str:
    plain = _strip_ansi(text)
    result: list[str] = []
    used = 0
    for char in plain:
        if unicodedata.combining(char):
            continue
        char_width = 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        if used + char_width > width:
            break
        result.append(char)
        used += char_width
    padding = max(0, width - used)
    return "".join(result) + (" " * padding)


def _wrap_display_text(text: str, width: int) -> list[str]:
    plain = _strip_ansi(text)
    if plain == "":
        return [""]

    lines: list[str] = []
    current: list[str] = []
    used = 0
    for char in plain:
        if char == "\n":
            lines.append("".join(current))
            current = []
            used = 0
            continue
        char_width = 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        if used and used + char_width > width:
            lines.append("".join(current))
            current = [char]
            used = char_width
            continue
        current.append(char)
        used += char_width
    lines.append("".join(current))
    return lines or [""]


def measure_renderable_lines(
    renderable: list[str] | Iterable[str] | object,
    *,
    color_system: str | None = "truecolor",
) -> int:
    if isinstance(renderable, list) and all(isinstance(line, str) for line in renderable):
        return len(renderable or [""])

    size = shutil.get_terminal_size(fallback=(96, 24))
    render_console = Console(
        force_terminal=True,
        color_system=color_system,
        legacy_windows=False,
        width=max(20, size.columns),
    )
    options = render_console.options.update(width=max(20, size.columns))
    rendered_lines = render_console.render_lines(renderable, options=options, pad=False, new_lines=False)
    return max(1, len(rendered_lines))


def _chat_frame_width() -> tuple[int, int]:
    columns = shutil.get_terminal_size(fallback=(96, 24)).columns
    indent = 0
    frame_width = max(24, columns)
    return frame_width, indent


def _frame_line(label: str, frame_width: int, indent: int, *, bottom: bool = False) -> str:
    if bottom:
        return (" " * indent) + (_FRAME_CHAR * frame_width)

    prefix = f"{_FRAME_CHAR * 2} {label} "
    fill = max(0, frame_width - _display_width(prefix))
    return (" " * indent) + prefix + (_FRAME_CHAR * fill)


def _is_tight_mode(cli_page_mode: str) -> bool:
    return str(cli_page_mode).strip().lower() == "tight"


def build_input_box_lines(
    lines: list[str],
    *,
    command_rows: list[tuple[str, str]] | None = None,
    selected_command: str | None = None,
    placeholder: str = "Type a message",
    multiline: bool = False,
    cli_page_mode: str = "loose",
) -> list[str]:
    frame_width, indent = _chat_frame_width()
    text_width = frame_width
    pad = " " * indent

    rendered = [_frame_line("Chat", frame_width, indent)]
    source_lines = lines or [""]

    for index, line in enumerate(source_lines):
        if index == 0 and not line:
            content = _fit_display_width(placeholder, text_width)
            rendered.append(f"{pad}{_DIM}{content}{_RESET}")
            continue

        wrapped = _wrap_display_text(line or "", text_width)
        for segment in wrapped:
            rendered.append(f"{pad}{_fit_display_width(segment, text_width)}")

    if command_rows:
        if not _is_tight_mode(cli_page_mode):
            rendered.append(f"{pad}{' ' * text_width}")
        rendered.append(f"{pad}{_DIM}{_fit_display_width('Commands', text_width)}{_RESET}")
        for name, description in command_rows:
            marker = "› " if name == selected_command else "  "
            row = _fit_display_width(f"{marker}{name:<9} {description}", text_width)
            if name == selected_command:
                rendered.append(f"{pad}{_BOLD}{row}{_RESET}")
            else:
                rendered.append(f"{pad}{_DIM}{row}{_RESET}")
    elif multiline:
        if not _is_tight_mode(cli_page_mode):
            rendered.append(f"{pad}{' ' * text_width}")
        hint = _fit_display_width('Multiline mode · End with """', text_width)
        rendered.append(f"{pad}{_DIM}{hint}{_RESET}")

    rendered.append(_frame_line("", frame_width, indent, bottom=True))
    return rendered


def build_effort_menu_lines(
    current: str,
    selected: str,
    *,
    supported: bool,
    cli_page_mode: str = "loose",
) -> list[str]:
    frame_width, indent = _chat_frame_width()
    text_width = frame_width
    pad = " " * indent
    lines = [_frame_line("Reasoning Effort", frame_width, indent)]

    for option in ("instant", "short", "default", "long"):
        marker = "› " if option == selected else "  "
        suffix = "  current" if option == current else ""
        content = _fit_display_width(f"{marker}{option}{suffix}", text_width)
        if option == selected:
            content = f"{_BOLD}{content}{_RESET}"
        elif option == current:
            content = f"{_DIM}{content}{_RESET}"
        lines.append(f"{pad}{content}")

    if not _is_tight_mode(cli_page_mode):
        lines.append(f"{pad}{' ' * text_width}")
    hint = _fit_display_width("Use ↑/↓ or j/k · Enter to apply · Esc to cancel", text_width)
    lines.append(f"{pad}{_DIM}{hint}{_RESET}")
    if not supported:
        note = _fit_display_width("This model may ignore effort controls.", text_width)
        lines.append(f"{pad}{_DIM}{note}{_RESET}")
    lines.append(_frame_line("", frame_width, indent, bottom=True))
    return lines


def build_cli_page_menu_lines(current: str, selected: str) -> list[str]:
    frame_width, indent = _chat_frame_width()
    text_width = frame_width
    pad = " " * indent
    lines = [_frame_line("CLI Page Mode", frame_width, indent)]

    labels = {
        "loose": "loose      Spacious default layout",
        "tight": "tight      Compact layout with minimal gaps",
    }
    for option in CLI_PAGE_MODES:
        marker = "› " if option == selected else "  "
        suffix = "  current" if option == current else ""
        content = _fit_display_width(f"{marker}{labels[option]}{suffix}", text_width)
        if option == selected:
            content = f"{_BOLD}{content}{_RESET}"
        elif option == current:
            content = f"{_DIM}{content}{_RESET}"
        lines.append(f"{pad}{content}")

    lines.append(f"{pad}{_DIM}{_fit_display_width('Use ↑/↓ or j/k · Enter to apply · Esc to cancel', text_width)}{_RESET}")
    lines.append(_frame_line("", frame_width, indent, bottom=True))
    return lines


def build_draft_menu_lines(current: str, selected: str) -> list[str]:
    frame_width, indent = _chat_frame_width()
    text_width = frame_width
    pad = " " * indent
    lines = [_frame_line("Draft Acceleration (Beta)", frame_width, indent)]

    labels = {
        "on": "on         Enable built-in Qwen3.5 draft acceleration",
        "off": "off        Disable draft acceleration",
    }
    for option in ("on", "off"):
        marker = "› " if option == selected else "  "
        suffix = "  current" if option == current else ""
        content = _fit_display_width(f"{marker}{labels[option]}{suffix}", text_width)
        if option == selected:
            content = f"{_BOLD}{content}{_RESET}"
        elif option == current:
            content = f"{_DIM}{content}{_RESET}"
        lines.append(f"{pad}{content}")

    lines.append(f"{pad}{_DIM}{_fit_display_width('Use ↑/↓ or j/k · Enter to apply · Esc to cancel', text_width)}{_RESET}")
    lines.append(_frame_line("", frame_width, indent, bottom=True))
    return lines


def _z_length_value(frame: int) -> float:
    """Animated cluster length in [0, 3] with smooth Z -> ZZ -> ZZZ -> blank transitions."""
    stage_values = (1.0, 2.0, 3.0, 0.0, 1.0)
    stage_count = len(stage_values) - 1
    phase = (frame % _Z_CYCLE) / _Z_CYCLE
    stage = min(stage_count - 1, int(phase * stage_count))
    stage_phase = (phase * stage_count) - stage
    eased = _smoothstep(stage_phase)
    start = stage_values[stage]
    end = stage_values[stage + 1]
    return start + (end - start) * eased


def _z_slot_brightness(frame: int, slot: int) -> float:
    """Brightness [0, 1] for a single Z slot in the evolving cluster."""
    length_value = _z_length_value(frame)
    activation = max(0.0, min(1.0, length_value - slot))
    if activation <= 0.02:
        return 0.0

    breathe_wave = 0.5 + 0.5 * math.sin((frame / 20.0 + slot * 0.13) * math.tau)
    breathe = 0.05 * _smoothstep(breathe_wave)
    shimmer_wave = 0.5 + 0.5 * math.sin((frame / 7.4) + slot * 1.07)
    shimmer = 0.03 * (_smoothstep(shimmer_wave) - 0.5)
    brightness = 0.14 + 0.42 * activation + breathe + shimmer
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


def choose_dream_word() -> str:
    """Pick the animated dream word with equal probability."""
    return random.choice(_DREAM_WORDS)


# ── Render functions ───────────────────────────────────────────────────

def render_daydreaming_text(
    frame: int = 0,
    *,
    rainbow: bool = False,
    label: str = "Daydreaming",
) -> Text:
    """Render a dreamy Z cluster, a sweeping glow word, and pulsing dots."""
    text = Text()
    text.append("  ")

    def dreamy_color(brightness: float, *, hue_position: float | None = None) -> str:
        level = max(0.0, min(1.0, brightness))
        if rainbow and hue_position is not None:
            color = _rainbow_color(hue_position, level)
        else:
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
            if rainbow:
                color = dreamy_color(char_brightness, hue_position=0.02 + slot * 0.08)
                text.append("Z", style=f"bold {color}" if char_brightness > 0.74 else color)
            else:
                text.append("Z", style=dreamy_style(char_brightness))
        else:
            text.append(" ")

    text.append("  ")

    # ── animated word with left-to-right sweep glow ──
    word = label
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
        color = dreamy_color(
            brightness,
            hue_position=0.12 + (idx / max(1, len(word) - 1)) * 0.72,
        )
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
        if rainbow:
            color = dreamy_color(dot_brightness, hue_position=0.86 + i * 0.05)
            text.append("·", style=color)
        else:
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


def render_reasoning_box(
    reasoning_text: str,
    frame: int,
    *,
    rainbow: bool = False,
    label: str = "Daydreaming",
    compact: bool = False,
) -> Group:
    """Render a framed reasoning box with animated header and clipped body."""
    normalized = reasoning_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if normalized.endswith("\n"):
        lines.append("")
    visible_count = 3 if compact else 5
    visible_lines = lines[-visible_count:] if lines else [""]
    if not any(line.strip() for line in visible_lines):
        visible_lines = [""] * max(0, visible_count - 1) + [" "]
    while len(visible_lines) < visible_count:
        visible_lines.insert(0, "")

    body = Text("\n".join(visible_lines), style=f"dim {_C_REASON}")
    return Group(
        render_frame_rule(render_daydreaming_text(frame, rainbow=rainbow, label=label)),
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
    reasoning_seconds: float | None = None,
    draft_active: bool = False,
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
    if reasoning_seconds is not None and reasoning_seconds > 0:
        text.append("  ·  ", style="dim")
        text.append(f"{reasoning_seconds:.1f}s thinking", style="dim")
    if draft_active:
        text.append("  ·  ", style="dim")
        text.append("draft", style="bold green")
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
    def __init__(self, console: Console, model_label: str, *, cli_page_mode: str = "loose", draft_active: bool = False):
        self.console = console
        self.model_label = model_label
        self.cli_page_mode = cli_page_mode
        self.draft_active = draft_active
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
        self._started_at: float | None = None

        # Threading
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._renderer: BottomTerminalRenderer | None = None
        self._lock = threading.Lock()
        self._dream_word = choose_dream_word()
        self._title_animator = TerminalTitleAnimator(self._dream_word, frames=_TITLE_Z_FRAMES)
        self._rainbow = random.random() < _RAINBOW_PROBABILITY
        self._stopped = False

    def _reasoning_time(self) -> float:
        elapsed = self._reasoning_elapsed or 0.0
        if self._reasoning_start is not None:
            return elapsed + (time.monotonic() - self._reasoning_start)
        return elapsed

    @property
    def had_reasoning(self) -> bool:
        return self._had_reasoning

    @property
    def reasoning_elapsed(self) -> float | None:
        value = self._reasoning_time()
        if value <= 0.0:
            return None
        return value

    def _render(self) -> Group:
        with self._lock:
            parts: list = []
            compact = _is_tight_mode(self.cli_page_mode)

            if self._reasoning_active:
                parts.append(render_reasoning_box(
                    self._reasoning_text,
                    self._frame,
                    rainbow=self._rainbow,
                    label=self._dream_word,
                    compact=compact,
                ))
                if not compact:
                    parts.append(Text())
            elif self._waiting and not self._had_reasoning:
                parts.append(
                    render_reasoning_box(
                        "",
                        self._frame,
                        rainbow=self._rainbow,
                        label=self._dream_word,
                        compact=compact,
                    )
                )
                if not compact:
                    parts.append(Text())
            else:
                # ── Reasoning summary (if model used thinking) ──
                if self._had_reasoning and self._reasoning_elapsed is not None:
                    parts.append(render_reasoning_line(self._reasoning_elapsed, active=False))
                    if not compact:
                        parts.append(Text())

                # ── Streamed output ──
                if self._output:
                    parts.append(Text(self._output))
                    if not compact:
                        parts.append(Text())

            # ── Footer (always at bottom) ──
            reasoning_hint = None
            if self._reasoning_active or self._had_reasoning:
                reasoning_hint = "Use /t to expand reasoning"
            live_reasoning = self._reasoning_time() if self._reasoning_active else None
            parts.append(render_status_footer(
                self.model_label,
                tokens_per_second=self._tokens_per_second,
                hint=reasoning_hint,
                reasoning_seconds=live_reasoning,
                draft_active=self.draft_active,
            ))

            return Group(*parts)

    def start(self) -> None:
        self._started_at = time.monotonic()
        self._title_animator.start()
        if not self.console.is_terminal:
            return
        self._renderer = BottomTerminalRenderer(
            self.console.file,
            clear_on_finish=True,
            color_system=self.console.color_system or "truecolor",
            scroll_on_grow=True,
            scroll_on_first_render=True,
            collapse_on_finish=_is_tight_mode(self.cli_page_mode),
        )
        self._renderer.render(self._render())
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(0.08):
            with self._lock:
                self._frame += 1
            if self._renderer is not None:
                self._renderer.render(self._render())

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
        if self._renderer is not None and not self._stopped:
            self._renderer.render(self._render())

    def ensure_minimum_wait(self, minimum_seconds: float) -> None:
        if self._started_at is None or self._reasoning_active:
            return
        elapsed = time.monotonic() - self._started_at
        remaining = minimum_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def start_reasoning(self) -> None:
        with self._lock:
            self._reasoning_active = True
            if self._reasoning_start is None:
                self._reasoning_start = time.monotonic()
            self._had_reasoning = True
            self._phase = "thinking"
        if self._renderer is not None and not self._stopped:
            self._renderer.render(self._render())

    def end_reasoning(self) -> None:
        with self._lock:
            self._reasoning_active = False
            if self._reasoning_start is not None:
                elapsed = time.monotonic() - self._reasoning_start
                self._reasoning_elapsed = (self._reasoning_elapsed or 0.0) + elapsed
                self._reasoning_start = None
        self._title_animator.stop()
        if self._renderer is not None and not self._stopped:
            self._renderer.render(self._render())

    def append_reasoning(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._reasoning_text += text
        if self._renderer is not None and not self._stopped:
            self._renderer.render(self._render())

    def append_output(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._output += text
        if self._renderer is not None and not self._stopped:
            self._renderer.render(self._render())

    def clear_output(self) -> None:
        with self._lock:
            self._output = ""
        if self._renderer is not None and not self._stopped:
            self._renderer.render(self._render())

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        if self._renderer is not None:
            self._renderer.render(self._render())
            self._renderer.finish()
            self._renderer = None
        self._title_animator.stop()


class BottomTerminalRenderer:
    def __init__(
        self,
        stream,
        *,
        clear_on_finish: bool = True,
        color_system: str | None = "truecolor",
        scroll_on_grow: bool = True,
        scroll_on_shrink: bool = False,
        scroll_on_first_render: bool = False,
        collapse_on_finish: bool = False,
    ) -> None:
        self.stream = stream
        self._clear_on_finish = clear_on_finish
        self._color_system = color_system
        self._scroll_on_grow = scroll_on_grow
        self._scroll_on_shrink = scroll_on_shrink
        self._scroll_on_first_render = scroll_on_first_render
        self._collapse_on_finish = collapse_on_finish
        self._line_count = 0
        self._cursor_hidden = False
        self._wrap_disabled = False
        self._start_row: int | None = None
        self._last_size = shutil.get_terminal_size(fallback=(96, 24))

    def _terminal_size(self):
        return shutil.get_terminal_size(fallback=(96, 24))

    def _move_to(self, row: int, col: int = 1) -> None:
        self.stream.write(f"\x1b[{row};{col}H")

    def _clear_from(self, row: int) -> None:
        self._move_to(max(1, row))
        self.stream.write("\x1b[J")

    def _scroll_up(self, count: int, *, rows: int) -> None:
        if count <= 0:
            return
        self._move_to(max(1, rows))
        self.stream.write("\n" * count)

    def _scroll_down(self, count: int) -> None:
        if count <= 0:
            return
        self._move_to(1)
        self.stream.write(f"\x1b[{count}L")

    def _renderable_to_lines(self, renderable: list[str] | Iterable[str] | object) -> list[str]:
        if isinstance(renderable, list) and all(isinstance(line, str) for line in renderable):
            return renderable or [""]

        size = self._terminal_size()
        render_console = Console(
            force_terminal=True,
            color_system=self._color_system,
            legacy_windows=False,
            width=max(20, size.columns),
        )
        options = render_console.options.update(width=max(20, size.columns))
        rendered_lines = render_console.render_lines(renderable, options=options, pad=False, new_lines=False)
        lines: list[str] = []
        for line in rendered_lines:
            lines.append(render_console._render_buffer(line))
        return lines or [""]

    def render(self, renderable: list[str] | Iterable[str] | object) -> None:
        if not self._cursor_hidden:
            self.stream.write("\x1b[?25l")
            self._cursor_hidden = True
        if not self._wrap_disabled:
            self.stream.write("\x1b[?7l")
            self._wrap_disabled = True

        lines = self._renderable_to_lines(renderable)
        size = self._terminal_size()
        start_row = max(1, size.lines - len(lines) + 1)
        first_render = self._start_row is None
        grows_upward = self._start_row is not None and start_row < self._start_row
        shrinks_downward = self._start_row is not None and start_row > self._start_row

        # When a bottom-pinned overlay needs more rows than the previous frame,
        # clear the old overlay first, then scroll blank space into view.
        # This keeps prior chat history in scrollback instead of letting the
        # growing overlay wipe lines that were already printed above it.
        if first_render:
            if self._scroll_on_first_render:
                self._scroll_up(len(lines), rows=size.lines)
            clear_from = start_row
        elif grows_upward:
            self._clear_from(self._start_row)
            if self._scroll_on_grow:
                self._scroll_up(self._start_row - start_row, rows=size.lines)
            clear_from = start_row
        elif shrinks_downward:
            self._clear_from(self._start_row)
            if self._scroll_on_shrink:
                self._scroll_down(start_row - self._start_row)
            clear_from = start_row
        else:
            clear_from = start_row if self._start_row is None else min(self._start_row, start_row)
        self._clear_from(clear_from)

        for index, line in enumerate(lines):
            self._move_to(start_row + index)
            self.stream.write("\x1b[2K")
            self.stream.write(line)

        self.stream.flush()
        self._start_row = start_row
        self._line_count = len(lines)
        self._last_size = size

    def wait_for_input(self, renderable: list[str] | Iterable[str] | object) -> None:
        fd = sys.stdin.fileno()
        while True:
            ready, _, _ = select.select([fd], [], [], 0.05)
            size = self._terminal_size()
            if size != self._last_size:
                self.render(renderable)
                continue
            if ready:
                return

    def finish(self) -> None:
        if self._line_count and self._start_row is not None and self._clear_on_finish:
            self._clear_from(self._start_row)
            self._move_to(self._start_row)
            if self._collapse_on_finish:
                self.stream.write(f"\x1b[{self._line_count}M")
                self._move_to(self._start_row)
        if self._cursor_hidden:
            self.stream.write("\x1b[?25h")
            self._cursor_hidden = False
        if self._wrap_disabled:
            self.stream.write("\x1b[?7h")
            self._wrap_disabled = False
        self.stream.flush()
        self._line_count = 0
        self._start_row = None


class InlineFlowRenderer(BottomTerminalRenderer):
    """Renderer anchored to the current transcript position instead of terminal bottom."""

    def __init__(
        self,
        stream,
        *,
        clear_on_finish: bool = True,
        color_system: str | None = "truecolor",
    ) -> None:
        super().__init__(stream, clear_on_finish=clear_on_finish, color_system=color_system)

    def _clear_inline_block(self, count: int) -> None:
        if count <= 0:
            return
        for index in range(count):
            self.stream.write("\r\x1b[2K")
            if index < count - 1:
                self.stream.write("\n")
        if count > 1:
            self.stream.write(f"\x1b[{count - 1}A")
        self.stream.write("\r")

    def _insert_inline_rows(self, count: int) -> None:
        if count <= 0:
            return
        self.stream.write(f"\x1b[{count}L")

    def _delete_inline_rows(self, count: int) -> None:
        if count <= 0:
            return
        self.stream.write(f"\x1b[{count}M")

    def render(self, renderable: list[str] | Iterable[str] | object) -> None:
        if not self._cursor_hidden:
            self.stream.write("\x1b[?25l")
            self._cursor_hidden = True
        if not self._wrap_disabled:
            self.stream.write("\x1b[?7l")
            self._wrap_disabled = True

        lines = self._renderable_to_lines(renderable)
        size = self._terminal_size()
        if self._line_count == 0:
            self._insert_inline_rows(len(lines))
        elif len(lines) > self._line_count:
            self.stream.write(f"\x1b[{self._line_count}B")
            self.stream.write("\r")
            self._insert_inline_rows(len(lines) - self._line_count)
            self.stream.write(f"\x1b[{self._line_count}A")
            self.stream.write("\r")
        elif len(lines) < self._line_count:
            self.stream.write(f"\x1b[{len(lines)}B")
            self.stream.write("\r")
            self._delete_inline_rows(self._line_count - len(lines))
            self.stream.write(f"\x1b[{len(lines)}A")
            self.stream.write("\r")

        clear_count = max(self._line_count, len(lines))
        self._clear_inline_block(clear_count)

        for index, line in enumerate(lines):
            self.stream.write("\r\x1b[2K")
            self.stream.write(line)
            if index < len(lines) - 1:
                self.stream.write("\n")

        if len(lines) > 1:
            self.stream.write(f"\x1b[{len(lines) - 1}A")
        self.stream.write("\r")

        self.stream.flush()
        self._line_count = len(lines)
        self._last_size = size

    def wait_for_input(self, renderable: list[str] | Iterable[str] | object) -> None:
        fd = sys.stdin.fileno()
        while True:
            ready, _, _ = select.select([fd], [], [], 0.05)
            size = self._terminal_size()
            if size != self._last_size:
                self.render(renderable)
                continue
            if ready:
                return

    def finish(self) -> None:
        if self._line_count and self._clear_on_finish:
            self._clear_inline_block(self._line_count)
            self._delete_inline_rows(self._line_count)
        if self._cursor_hidden:
            self.stream.write("\x1b[?25h")
            self._cursor_hidden = False
        if self._wrap_disabled:
            self.stream.write("\x1b[?7h")
            self._wrap_disabled = False
        self.stream.flush()
        self._line_count = 0


@contextmanager
def daydreaming_status(console: Console, model_label: str, *, cli_page_mode: str = "loose", draft_active: bool = False):
    status = ConversationStatus(console, model_label, cli_page_mode=cli_page_mode, draft_active=draft_active)
    status.start()
    try:
        yield status
    finally:
        status.stop()


def build_dreaming_menu_lines(selected: str, *, cli_page_mode: str = "loose") -> list[str]:
    """Arrow-key menu for /dreaming mode selection."""
    frame_width, indent = _chat_frame_width()
    text_width = frame_width
    pad = " " * indent
    lines = [_frame_line("Dreaming Mode", frame_width, indent)]

    options = [
        ("reming", "REMing", "Quick memory extraction"),
        ("daydream", "Daydream", "Full sleep cycle (N3\u2192N2\u2192REM)"),
    ]
    for key, label, desc in options:
        marker = "\u203a " if key == selected else "  "
        content = _fit_display_width(f"{marker}{label:<12} {desc}", text_width)
        if key == selected:
            content = f"{_BOLD}{content}{_RESET}"
        else:
            content = f"{_DIM}{content}{_RESET}"
        lines.append(f"{pad}{content}")

    if not _is_tight_mode(cli_page_mode):
        lines.append(f"{pad}{' ' * text_width}")
    hint = _fit_display_width("Use \u2191/\u2193 or j/k \u00b7 Enter to start \u00b7 Esc to cancel", text_width)
    lines.append(f"{pad}{_DIM}{hint}{_RESET}")
    lines.append(_frame_line("", frame_width, indent, bottom=True))
    return lines


def build_session_list_lines(
    sessions: list,
    selected_index: int,
    *,
    allow_delete: bool = False,
    cli_page_mode: str = "loose",
) -> list[str]:
    """Arrow-key menu for /resume session selection."""
    frame_width, indent = _chat_frame_width()
    text_width = frame_width
    pad = " " * indent
    lines = [_frame_line("Resume Session", frame_width, indent)]

    visible = sessions[:10]
    for i, session in enumerate(visible):
        marker = "\u203a " if i == selected_index else "  "
        title = session.title or "(untitled)"
        if len(title) > 30:
            title = title[:27] + "..."
        model = session.model or ""
        msg_count = len(session.messages)
        date_str = format_time_ago(session.updated_at) if session.updated_at else ""
        row = f"{marker}{title:<32} {model:<12} {date_str:<14} {msg_count} msgs"
        content = _fit_display_width(row, text_width)
        if i == selected_index:
            content = f"{_BOLD}{content}{_RESET}"
        else:
            content = f"{_DIM}{content}{_RESET}"
        lines.append(f"{pad}{content}")

    if not _is_tight_mode(cli_page_mode):
        lines.append(f"{pad}{' ' * text_width}")
    hint_text = "Use \u2191/\u2193 or j/k \u00b7 Enter to resume"
    if allow_delete:
        hint_text += " \u00b7 d to delete"
    hint_text += " \u00b7 Esc to cancel"
    hint = _fit_display_width(hint_text, text_width)
    lines.append(f"{pad}{_DIM}{hint}{_RESET}")
    lines.append(_frame_line("", frame_width, indent, bottom=True))
    return lines


def build_memory_display_lines(memories: list) -> list[str]:
    """Display for /memory command."""
    frame_width, indent = _chat_frame_width()
    text_width = frame_width
    pad = " " * indent
    lines = [_frame_line("Session Memories", frame_width, indent)]

    category_icons = {
        "fact": "[F]",
        "preference": "[P]",
        "context": "[C]",
        "insight": "[I]",
        "pattern": "[X]",
    }

    for mem in memories:
        icon = category_icons.get(mem.category, "[?]")
        importance = mem.importance if hasattr(mem, "importance") else 0.5
        bar_len = int(importance * 10)
        bar = "\u2588" * bar_len + "\u2591" * (10 - bar_len)
        row = f"  {icon} {mem.content}"
        if len(row) > text_width - 14:
            row = row[:text_width - 17] + "..."
        content = _fit_display_width(f"{row:<{text_width - 13}} {bar}", text_width)
        lines.append(f"{pad}{_DIM}{content}{_RESET}")

    if not memories:
        content = _fit_display_width("  No memories extracted yet.", text_width)
        lines.append(f"{pad}{_DIM}{content}{_RESET}")

    lines.append(_frame_line("", frame_width, indent, bottom=True))
    return lines


def render_sleep_phase_box(
    phase: str,
    text: str,
    frame: int,
    *,
    label: str | None = None,
) -> Group:
    """Animated box during dreaming phases.

    Similar to render_reasoning_box() but uses SLEEP_PHASE_PALETTES[phase].
    Shows phase label + streamed model output in a dim scrolling box.
    """
    palette = SLEEP_PHASE_PALETTES.get(phase, (_C_MID, _C_BRIGHT, _C_GLOW))
    dark, mid, glow = palette
    header_label = label or REFLECTING_LABEL
    phase_label = SLEEP_PHASE_LABELS.get(phase, phase)

    header = render_daydreaming_text(frame, label=header_label)
    header.append("  ")
    header.append(phase_label, style=f"bold {mid}")

    # Body: last 5 lines of output
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    body_lines = normalized.split("\n")
    if normalized.endswith("\n"):
        body_lines.append("")
    visible = body_lines[-5:] if body_lines else [""]
    if not any(line.strip() for line in visible):
        visible = [""] * 4 + [" "]
    while len(visible) < 5:
        visible.insert(0, "")

    body = Text("\n".join(visible), style=f"dim {dark}")

    return Group(
        render_frame_rule(header),
        body,
        render_frame_rule(),
    )


class DreamingStatus:
    """Thread-safe animated status for /dreaming.

    Shows current phase label with phase-colored Z animation,
    scrolling box of model output, and phase progress indicator.
    Uses BottomTerminalRenderer.
    """

    def __init__(self, console: Console, model_label: str):
        self.console = console
        self.model_label = model_label
        self._frame = 0
        self._phase = "reming"
        self._phase_text = ""
        self._status_text = ""
        self._phase_number = 0
        self._total_phases = 1

        # Threading
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._renderer: BottomTerminalRenderer | None = None
        self._lock = threading.Lock()
        self._title_animator = TerminalTitleAnimator(REFLECTING_LABEL, frames=_TITLE_Z_FRAMES)

    def _render(self) -> Group:
        with self._lock:
            parts: list = []

            parts.append(render_sleep_phase_box(
                self._phase,
                self._phase_text,
                self._frame,
                label=REFLECTING_LABEL,
            ))
            parts.append(Text())

            phase_label = SLEEP_PHASE_LABELS.get(self._phase, self._phase)
            if self._total_phases > 1 and self._phase_number:
                phase_footer = f"{phase_label}  {self._phase_number}/{self._total_phases}"
            else:
                phase_footer = phase_label

            footer = render_status_footer(
                self.model_label,
                phase=phase_footer,
                hint=self._status_text or None,
            )
            parts.append(footer)

            return Group(*parts)

    def start(self) -> None:
        self._title_animator.start()
        if not self.console.is_terminal:
            return
        self._renderer = BottomTerminalRenderer(
            self.console.file,
            clear_on_finish=True,
            color_system=self.console.color_system or "truecolor",
        )
        self._renderer.render(self._render())
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(0.08):
            with self._lock:
                self._frame += 1
            if self._renderer is not None:
                self._renderer.render(self._render())

    def set_phase(self, name: str, status_text: str = "", *, number: int = 0, total: int = 1) -> None:
        with self._lock:
            self._phase = name
            self._status_text = status_text
            self._phase_text = ""
            self._phase_number = number
            self._total_phases = total
        if self._renderer is not None:
            self._renderer.render(self._render())

    def append_text(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._phase_text += text
        if self._renderer is not None:
            self._renderer.render(self._render())

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        if self._renderer is not None:
            self._renderer.render(self._render())
            self._renderer.finish()
        self._title_animator.stop()


@contextmanager
def dreaming_status(console: Console, model_label: str):
    """Context manager for DreamingStatus."""
    status = DreamingStatus(console, model_label)
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

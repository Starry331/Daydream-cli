from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DAYDREAM_HOME = Path(os.environ.get("DAYDREAM_HOME", "~/.daydream")).expanduser()
REGISTRY_FILE = DAYDREAM_HOME / "registry.yaml"
CONFIG_FILE = DAYDREAM_HOME / "config.yaml"
PROFILES_FILE = DAYDREAM_HOME / "profiles.yaml"
SERVER_STATE_FILE = DAYDREAM_HOME / "server.json"
SERVER_LOG_FILE = DAYDREAM_HOME / "server.log"
CHATS_DIR = DAYDREAM_HOME / "chats"
MEMORIES_DIR = DAYDREAM_HOME / "memories"

HF_HOME = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
MODEL_CACHE_DIR = Path(
    os.environ.get(
        "DAYDREAM_CACHE_DIR",
        os.environ.get("HF_HUB_CACHE", str(HF_HOME / "hub")),
    )
).expanduser()
LOCAL_MODELS_DIR = Path(
    os.environ.get("DAYDREAM_LOCAL_MODELS_DIR", str(DAYDREAM_HOME / "models"))
).expanduser()

DEFAULT_MODEL = "qwen3:8b"
DEFAULT_TEMP = 0.6
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 4096
DEFAULT_PORT = 11434
DEFAULT_HOST = "127.0.0.1"
DEFAULT_CLI_PAGE_MODE = "loose"

# Context length presets. "auto" lets Daydream pick a safe value based
# on the model's max_position_embeddings and the GPU's working-set
# budget. "custom" is anything else — capped at a safety ceiling on
# load so a too-high persisted value can't lock the user out of
# launching daydream again.
DEFAULT_CONTEXT_LENGTH = "auto"
CONTEXT_LENGTH_PRESETS: tuple[tuple[str, int | None], ...] = (
    ("auto", None),
    ("4k",     4 * 1024),
    ("8k",     8 * 1024),
    ("16k",   16 * 1024),
    ("32k",   32 * 1024),
    ("64k",   64 * 1024),
    ("128k", 128 * 1024),
)
# Hard upper bound. Any persisted/custom value above this is clamped on
# load — protects against “I typed 1000000 once and now daydream
# crashes every time I open it”.
CONTEXT_LENGTH_HARD_CEILING = 256 * 1024


def ensure_home() -> None:
    """Create ~/.daydream and subdirectories if they don't exist."""
    DAYDREAM_HOME.mkdir(parents=True, exist_ok=True)
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    MEMORIES_DIR.mkdir(parents=True, exist_ok=True)


def _load_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}

    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return {}

    return data if isinstance(data, dict) else {}


def _write_config(config: dict[str, Any]) -> None:
    ensure_home()
    with CONFIG_FILE.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def _get_nested(config: dict[str, Any], *keys: str) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _coerce_str(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value)


def _coerce_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_default_model() -> str:
    return _coerce_str(_load_config().get("model"), DEFAULT_MODEL)


def get_default_temp() -> float:
    config = _load_config()
    return _coerce_float(_get_nested(config, "run", "temp"), DEFAULT_TEMP)


def get_default_top_p() -> float:
    config = _load_config()
    return _coerce_float(_get_nested(config, "run", "top_p"), DEFAULT_TOP_P)


def get_default_max_tokens() -> int:
    config = _load_config()
    return _coerce_int(_get_nested(config, "run", "max_tokens"), DEFAULT_MAX_TOKENS)


def get_default_host() -> str:
    config = _load_config()
    return _coerce_str(_get_nested(config, "serve", "host"), DEFAULT_HOST)


def get_default_port() -> int:
    config = _load_config()
    return _coerce_int(_get_nested(config, "serve", "port"), DEFAULT_PORT)


def get_default_cli_page_mode() -> str:
    config = _load_config()
    value = _coerce_str(_get_nested(config, "chat", "cli_page_mode"), DEFAULT_CLI_PAGE_MODE).strip().lower()
    if value in {"tight", "loose"}:
        return value
    return DEFAULT_CLI_PAGE_MODE


def set_default_cli_page_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in {"tight", "loose"}:
        normalized = DEFAULT_CLI_PAGE_MODE

    config = _load_config()
    chat = config.get("chat")
    if not isinstance(chat, dict):
        chat = {}
        config["chat"] = chat
    chat["cli_page_mode"] = normalized
    _write_config(config)
    return normalized


def get_default_context_length() -> str:
    """Return the user's preferred context-length preset.

    Values: "auto" | "4k" | "8k" | "16k" | "32k" | "64k" | "128k" |
    a stringified integer (custom). The value is *not* sanitized
    against a model — `resolve_context_length` does that at run time.
    """
    config = _load_config()
    value = _get_nested(config, "run", "context_length")
    if value is None:
        return DEFAULT_CONTEXT_LENGTH
    text = str(value).strip().lower()
    if not text:
        return DEFAULT_CONTEXT_LENGTH
    return text


def set_default_context_length(value: str) -> str:
    """Persist the context-length preset. Returns the normalized value."""
    normalized = _normalize_context_length(value)
    config = _load_config()
    run = config.get("run")
    if not isinstance(run, dict):
        run = {}
        config["run"] = run
    run["context_length"] = normalized
    _write_config(config)
    return normalized


def _normalize_context_length(value: str | int | None) -> str:
    """Normalize a context-length preference string."""
    if value is None:
        return DEFAULT_CONTEXT_LENGTH
    if isinstance(value, int):
        return _clamp_context_int(value)
    text = str(value).strip().lower()
    if not text or text == "auto" or text == "default":
        return "auto"
    for name, _ in CONTEXT_LENGTH_PRESETS:
        if text == name:
            return name
    # Try to parse as int; accept "32k" / "32K" / "32 000" / "32768".
    if text.endswith("k"):
        try:
            return _clamp_context_int(int(float(text[:-1]) * 1024))
        except ValueError:
            return DEFAULT_CONTEXT_LENGTH
    if text.endswith("m"):
        try:
            return _clamp_context_int(int(float(text[:-1]) * 1024 * 1024))
        except ValueError:
            return DEFAULT_CONTEXT_LENGTH
    try:
        return _clamp_context_int(int(text.replace(",", "").replace(" ", "")))
    except ValueError:
        return DEFAULT_CONTEXT_LENGTH


def _clamp_context_int(value: int) -> str:
    """Clamp a custom integer to the hard ceiling and return a string."""
    if value <= 0:
        return DEFAULT_CONTEXT_LENGTH
    if value > CONTEXT_LENGTH_HARD_CEILING:
        value = CONTEXT_LENGTH_HARD_CEILING
    return str(value)


def resolve_context_length(
    preference: str | None,
    *,
    model_max_position_embeddings: int | None = None,
) -> int | None:
    """Translate a stored preference into an actual integer ctx length.

    Returns None when the preference is "auto" — the caller should
    fall back to the model's own default (mlx-lm uses
    max_position_embeddings).

    Always clamps to the hard ceiling AND to the model's own
    max_position_embeddings when known, even for explicit presets, so
    we never hand a value to mlx-lm that the kernel can't honour.
    """
    pref = _normalize_context_length(preference)
    if pref == "auto":
        return None
    for name, n in CONTEXT_LENGTH_PRESETS:
        if pref == name:
            value = n
            break
    else:
        try:
            value = int(pref)
        except ValueError:
            return None
    if value is None:
        return None
    if value > CONTEXT_LENGTH_HARD_CEILING:
        value = CONTEXT_LENGTH_HARD_CEILING
    if model_max_position_embeddings and value > model_max_position_embeddings:
        value = model_max_position_embeddings
    return value


def get_local_model_roots() -> list[Path]:
    config = _load_config()
    configured = _get_nested(config, "models", "local_roots")
    roots: list[Path] = [LOCAL_MODELS_DIR]

    env_roots = os.environ.get("DAYDREAM_MODELS_DIRS")
    if env_roots:
        roots.extend(Path(part).expanduser() for part in env_roots.split(os.pathsep) if part.strip())

    if isinstance(configured, list):
        roots.extend(Path(str(part)).expanduser() for part in configured if str(part).strip())

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique

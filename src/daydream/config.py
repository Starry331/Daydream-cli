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

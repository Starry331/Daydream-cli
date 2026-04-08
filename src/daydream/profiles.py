"""Custom Daydream model profiles, similar to a small Ollama Modelfile."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel

from daydream.config import PROFILES_FILE, ensure_home
from daydream.registry import list_available

console = Console()

PROFILE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SUPPORTED_PARAMETERS = {"temperature", "top_p", "max_tokens", "effort"}
EFFORT_LEVELS = ("instant", "short", "default", "long")


@dataclass(frozen=True)
class ModelProfile:
    name: str
    from_model: str
    system: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


def normalize_profile_name(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "-")
    if not PROFILE_NAME_RE.match(normalized):
        raise ValueError(
            "Profile names may only contain lowercase letters, numbers, '.', '_' or '-'."
        )
    return normalized


def _canonical_parameter_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "temp": "temperature",
        "temperature": "temperature",
        "top_p": "top_p",
        "topp": "top_p",
        "max_tokens": "max_tokens",
        "maxtokens": "max_tokens",
        "effort": "effort",
    }
    result = aliases.get(normalized)
    if result is None or result not in SUPPORTED_PARAMETERS:
        raise ValueError(f"Unsupported PARAMETER '{name}'.")
    return result


def _parse_parameter_value(name: str, value: str) -> Any:
    if name == "temperature" or name == "top_p":
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"PARAMETER {name} expects a number.") from exc
    if name == "max_tokens":
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError("PARAMETER max_tokens expects an integer.") from exc
    if name == "effort":
        effort = value.strip().lower()
        if effort not in EFFORT_LEVELS:
            allowed = ", ".join(EFFORT_LEVELS)
            raise ValueError(f"PARAMETER effort must be one of: {allowed}.")
        return effort
    raise ValueError(f"Unsupported PARAMETER '{name}'.")


def _validate_profile_name_available(name: str) -> None:
    builtin_names = {
        family if variant == "default" else f"{family}:{variant}"
        for family, variant, _ in list_available()
    }
    if name in builtin_names:
        raise ValueError(
            f"Profile name '{name}' conflicts with an existing model alias. Pick a different name."
        )


def _load_profile_map() -> dict[str, dict[str, Any]]:
    if not PROFILES_FILE.exists():
        return {}
    try:
        data = yaml.safe_load(PROFILES_FILE.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for name, payload in data.items():
        if not isinstance(name, str) or not isinstance(payload, dict):
            continue
        result[name.lower()] = dict(payload)
    return result


def _save_profile_map(data: dict[str, dict[str, Any]]) -> None:
    ensure_home()
    PROFILES_FILE.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")


def _profile_from_payload(name: str, payload: dict[str, Any]) -> ModelProfile:
    from_model = str(payload.get("from") or "").strip()
    if not from_model:
        raise ValueError(f"Profile '{name}' is missing FROM.")
    parameters = payload.get("parameters") or {}
    if not isinstance(parameters, dict):
        parameters = {}
    normalized_parameters = {
        _canonical_parameter_name(str(key)): _parse_parameter_value(
            _canonical_parameter_name(str(key)),
            str(value),
        )
        for key, value in parameters.items()
    }
    system = payload.get("system")
    if system is not None:
        system = str(system)
    return ModelProfile(
        name=name,
        from_model=from_model,
        system=system,
        parameters=normalized_parameters,
    )


def list_profiles() -> list[ModelProfile]:
    profiles: list[ModelProfile] = []
    for name, payload in sorted(_load_profile_map().items()):
        try:
            profiles.append(_profile_from_payload(name, payload))
        except ValueError:
            continue
    return profiles


def get_profile(name: str) -> ModelProfile | None:
    normalized = name.strip().lower()
    payload = _load_profile_map().get(normalized)
    if payload is None:
        return None
    return _profile_from_payload(normalized, payload)


def save_profile(profile: ModelProfile) -> None:
    name = normalize_profile_name(profile.name)
    _validate_profile_name_available(name)
    data = _load_profile_map()
    data[name] = {
        "from": profile.from_model,
        "system": profile.system,
        "parameters": dict(profile.parameters),
    }
    _save_profile_map(data)


def delete_profile(name: str) -> bool:
    normalized = name.strip().lower()
    data = _load_profile_map()
    removed = data.pop(normalized, None)
    if removed is None:
        return False
    _save_profile_map(data)
    return True


def parse_daydreamfile(path: str | Path, *, name: str) -> ModelProfile:
    profile_name = normalize_profile_name(name)
    _validate_profile_name_available(profile_name)
    raw_lines = Path(path).read_text(encoding="utf-8").splitlines()
    from_model: str | None = None
    system: str | None = None
    parameters: dict[str, Any] = {}
    index = 0

    while index < len(raw_lines):
        raw_line = raw_lines[index]
        line = raw_line.strip()
        index += 1
        if not line or line.startswith("#"):
            continue

        keyword, _, remainder = line.partition(" ")
        directive = keyword.upper()
        value = remainder.strip()

        if directive == "FROM":
            if not value:
                raise ValueError("FROM requires a base model reference.")
            from_model = value
            continue

        if directive == "SYSTEM":
            if value.startswith('"""'):
                current = value[3:]
                collected: list[str] = []
                if current.endswith('"""'):
                    collected.append(current[:-3])
                else:
                    if current:
                        collected.append(current)
                    while index < len(raw_lines):
                        next_line = raw_lines[index]
                        index += 1
                        if next_line.endswith('"""'):
                            collected.append(next_line[:-3])
                            break
                        collected.append(next_line)
                    else:
                        raise ValueError("Unterminated SYSTEM block. Close it with triple quotes.")
                system = "\n".join(collected).strip() or None
            else:
                system = value or None
            continue

        if directive == "PARAMETER":
            param_name, _, param_value = value.partition(" ")
            if not param_name or not param_value.strip():
                raise ValueError("PARAMETER requires a name and a value.")
            canonical = _canonical_parameter_name(param_name)
            parameters[canonical] = _parse_parameter_value(canonical, param_value.strip())
            continue

        raise ValueError(f"Unsupported directive '{keyword}'.")

    if not from_model:
        raise ValueError("Daydreamfile is missing FROM.")

    return ModelProfile(
        name=profile_name,
        from_model=from_model,
        system=system,
        parameters=parameters,
    )


def create_profile(name: str, *, file_path: str | Path) -> ModelProfile:
    profile = parse_daydreamfile(file_path, name=name)
    save_profile(profile)
    return profile


def show_profile(name: str) -> None:
    profile = get_profile(name)
    if profile is None:
        raise ValueError(f"Unknown custom model '{name}'.")

    info_lines = [
        f"[bold]Custom model:[/]    {profile.name}",
        f"[bold]Base model:[/]      {profile.from_model}",
    ]
    if profile.system:
        info_lines.append("")
        info_lines.append("[bold]System prompt:[/]")
        info_lines.append(profile.system)
    if profile.parameters:
        info_lines.append("")
        info_lines.append("[bold]Parameters:[/]")
        for key, value in profile.parameters.items():
            info_lines.append(f"- {key}: {value}")
    else:
        info_lines.append("")
        info_lines.append("[dim]No custom parameters.[/dim]")

    console.print(Panel("\n".join(info_lines), title=f"[bold]{profile.name}[/]", border_style="cyan"))

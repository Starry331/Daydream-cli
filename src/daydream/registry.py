"""Model name registry — maps short names to HuggingFace repo IDs."""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional

import yaml

from daydream.config import REGISTRY_FILE, ensure_home, get_local_model_roots

# Format: family -> { variant -> HF repo ID }
# "default" variant is used when no variant is specified.
BUILTIN_REGISTRY: dict[str, dict[str, str]] = {
    "qwen3": {
        "default": "mlx-community/Qwen3-8B-4bit",
        "0.6b": "mlx-community/Qwen3-0.6B-4bit",
        "1.7b": "mlx-community/Qwen3-1.7B-4bit",
        "4b": "mlx-community/Qwen3-4B-4bit",
        "8b": "mlx-community/Qwen3-8B-4bit",
        "14b": "mlx-community/Qwen3-14B-4bit",
        "30b": "mlx-community/Qwen3-30B-A3B-4bit",
        "32b": "mlx-community/Qwen3-32B-4bit",
    },
    "qwen2.5-coder": {
        "default": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "1.5b": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
        "3b": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
        "7b": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "14b": "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
        "32b": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
    },
    "llama3.2": {
        "default": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "1b": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "3b": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    },
    "llama4": {
        "default": "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
        "scout": "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
    },
    "gemma3": {
        "default": "mlx-community/gemma-3-4b-it-4bit",
        "1b": "mlx-community/gemma-3-1b-it-4bit",
        "4b": "mlx-community/gemma-3-4b-it-4bit",
        "12b": "mlx-community/gemma-3-12b-it-4bit",
        "27b": "mlx-community/gemma-3-27b-it-4bit",
    },
    "deepseek-r1": {
        "default": "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit",
        "8b": "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit",
        "14b": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
        "32b": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
    },
    "phi-4": {
        "default": "mlx-community/phi-4-4bit",
        "14b": "mlx-community/phi-4-4bit",
        "mini": "mlx-community/phi-4-mini-instruct-4bit",
    },
    "mistral": {
        "default": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "7b": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "nemo": "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
        "small": "mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
    },
    "codellama": {
        "default": "mlx-community/CodeLlama-7b-Instruct-hf-4bit-MLX",
        "7b": "mlx-community/CodeLlama-7b-Instruct-hf-4bit-MLX",
        "13b": "mlx-community/CodeLlama-13b-Instruct-hf-4bit-MLX",
    },
    "smollm2": {
        "default": "mlx-community/SmolLM2-1.7B-Instruct-4bit",
        "135m": "mlx-community/SmolLM2-135M-Instruct-4bit",
        "360m": "mlx-community/SmolLM2-360M-Instruct-4bit",
        "1.7b": "mlx-community/SmolLM2-1.7B-Instruct-4bit",
    },
    "devstral": {
        "default": "mlx-community/Devstral-Small-2505-4bit",
        "small": "mlx-community/Devstral-Small-2505-4bit",
    },
}

MODEL_CONFIG_FILES = ("config.json",)
MODEL_TOKENIZER_FILES = (
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
)
MODEL_WEIGHT_FILES = (
    "*.safetensors",
    "model.safetensors.index.json",
)


def _save_user_registry(data: dict[str, dict[str, str]]) -> None:
    ensure_home()
    with REGISTRY_FILE.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True)


def _format_short_name(family: str, variant: str, variants: dict[str, str]) -> Optional[str]:
    if variant == "default":
        non_default_variants = [name for name in variants if name != "default"]
        if not non_default_variants:
            return family
        return None
    return f"{family}:{variant}"


def _looks_like_local_path(name: str) -> bool:
    return (
        name.startswith(("~", ".", "/"))
        or name.startswith(f"..{os.sep}")
        or f"{os.sep}" in name
    )


def _is_local_target(value: str) -> bool:
    return value.startswith(("~", ".", "/"))


def normalize_hf_reference(name: str) -> str:
    value = name.strip()
    if value.startswith("hf.co/"):
        parts = [part for part in value[6:].split("/") if part]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return value

    if value.startswith(("https://hf.co/", "http://hf.co/", "https://huggingface.co/", "http://huggingface.co/")):
        parsed = urlparse(value)
        parts = [part for part in parsed.path.split("/") if part]
        if parts[:1] == ["models"]:
            parts = parts[1:]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    return value


def _sanitize_alias(name: str) -> str:
    alias = name.strip().lower().replace("_", "-").replace(" ", "-")
    alias = re.sub(r"[^a-z0-9.\-]+", "-", alias)
    alias = re.sub(r"-+", "-", alias).strip("-")

    stripped = True
    while stripped and alias:
        stripped = False
        for suffix in ("-mlx", "-4bit", "-8bit", "-fp16", "-bf16", "-int4", "-int8"):
            if alias.endswith(suffix):
                alias = alias[: -len(suffix)].rstrip("-")
                stripped = True

    return alias or "local-model"


def is_local_model_dir(path: Path) -> bool:
    path = path.expanduser()
    if not path.is_dir():
        return False

    has_config = all((path / file_name).exists() for file_name in MODEL_CONFIG_FILES)
    if not has_config:
        return False

    has_tokenizer = any((path / file_name).exists() for file_name in MODEL_TOKENIZER_FILES)
    has_tokenizer = has_tokenizer or any(path.glob("*.tiktoken"))
    has_fixture = (path / "daydream_fixture.json").exists()

    has_weights = (path / "model.safetensors.index.json").exists() or any(path.glob("*.safetensors"))
    return (has_tokenizer and has_weights) or has_fixture


def _iter_model_dirs(root: Path, *, max_depth: int = 3) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []

    results: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    seen: set[Path] = set()

    while stack:
        current, depth = stack.pop()
        if current in seen:
            continue
        seen.add(current)

        if is_local_model_dir(current):
            results.append(current.resolve())
            continue

        if depth >= max_depth:
            continue

        try:
            children = [child for child in current.iterdir() if child.is_dir()]
        except OSError:
            continue

        for child in children:
            stack.append((child, depth + 1))

    return sorted(results)


def _find_short_name_for_target(
    target: str,
    registry: Optional[dict[str, dict[str, str]]] = None,
) -> Optional[str]:
    registry = registry or _get_merged_registry()
    for family, variants in registry.items():
        for variant, value in variants.items():
            if value != target:
                continue
            short_name = _format_short_name(family, variant, variants)
            if short_name:
                return short_name
    return None


def register_local_model(path: str | Path) -> str:
    model_dir = Path(path).expanduser().resolve()
    if not is_local_model_dir(model_dir):
        raise ValueError(f"Not a valid local MLX model directory: {model_dir}")

    target = str(model_dir)
    existing = _find_short_name_for_target(target)
    if existing:
        return existing

    user_registry = _load_user_registry()
    family = _sanitize_alias(model_dir.name)
    existing_registry = _get_merged_registry()

    if family in existing_registry:
        variant_target = existing_registry[family].get("default")
        if variant_target == target and len(existing_registry[family]) == 1:
            return family

        index = 2
        while f"{family}-{index}" in existing_registry:
            index += 1
        family = f"{family}-{index}"

    user_registry[family] = {"default": target}
    _save_user_registry(user_registry)
    return family


def register_remote_model(repo_id: str) -> str:
    target = normalize_hf_reference(repo_id)
    existing = _find_short_name_for_target(target)
    if existing:
        return existing

    user_registry = _load_user_registry()
    family = _sanitize_alias(target.split("/")[-1])
    existing_registry = _get_merged_registry()

    if family in existing_registry:
        variant_target = existing_registry[family].get("default")
        if variant_target == target and len(existing_registry[family]) == 1:
            return family

        index = 2
        while f"{family}-{index}" in existing_registry:
            index += 1
        family = f"{family}-{index}"

    user_registry[family] = {"default": target}
    _save_user_registry(user_registry)
    return family


def scan_local_models(*, persist: bool = True) -> list[tuple[str, str]]:
    discovered: list[tuple[str, str]] = []
    seen_targets: set[str] = set()

    for root in get_local_model_roots():
        for model_dir in _iter_model_dirs(root):
            target = str(model_dir)
            if target in seen_targets:
                continue
            seen_targets.add(target)

            if persist:
                short_name = register_local_model(model_dir)
            else:
                short_name = _find_short_name_for_target(target) or _sanitize_alias(model_dir.name)

            discovered.append((short_name, target))

    return discovered


def _load_user_registry() -> dict[str, dict[str, str]]:
    """Load user overrides from ~/.daydream/registry.yaml."""
    if not REGISTRY_FILE.exists():
        return {}
    try:
        with REGISTRY_FILE.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return {}
        normalized: dict[str, dict[str, str]] = {}
        for family, variants in data.items():
            if not isinstance(family, str) or not isinstance(variants, dict):
                continue
            normalized[family.lower()] = {
                str(variant).lower(): str(repo_id)
                for variant, repo_id in variants.items()
                if isinstance(repo_id, str)
            }
        return normalized
    except Exception:
        return {}


def _get_merged_registry() -> dict[str, dict[str, str]]:
    """Merge built-in and user registries."""
    merged = {}
    for family, variants in BUILTIN_REGISTRY.items():
        merged[family] = dict(variants)
    for family, variants in _load_user_registry().items():
        if family in merged:
            merged[family].update(variants)
        else:
            merged[family] = dict(variants)
    return merged


def resolve(name: str) -> str:
    """Resolve a short model name to a HuggingFace repo ID.

    Accepts:
        "qwen3:8b"                     -> lookup in registry
        "qwen3"                        -> lookup "default" variant
        "mlx-community/Qwen3-8B-4bit"  -> passthrough (contains '/')
    """
    name = normalize_hf_reference(name.strip())
    if not name:
        raise ValueError("Model name cannot be empty.")

    if _looks_like_local_path(name):
        local_path = Path(name).expanduser()
        if local_path.exists():
            if not is_local_model_dir(local_path):
                raise ValueError(f"Not a valid local MLX model directory: {local_path}")
            register_local_model(local_path)
            return str(local_path.resolve())
        if name.startswith(("~", ".", "/")):
            raise ValueError(f"Local model path not found: {local_path}")

    # Passthrough for full HF repo IDs
    if "/" in name:
        return name

    registry = _get_merged_registry()

    # Parse family:variant
    if ":" in name:
        family, variant = name.split(":", 1)
    else:
        family, variant = name, "default"

    family = family.lower()
    variant = (variant or "default").lower()

    if family not in registry:
        scan_local_models(persist=True)
        registry = _get_merged_registry()

    if family not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Unknown model family '{family}'. "
            f"Available: {available}\n"
            f"Or pass a full HuggingFace repo ID or local model path"
        )

    variants = registry[family]
    if variant not in variants:
        available = ", ".join(v for v in sorted(variants.keys()) if v != "default")
        raise ValueError(
            f"Unknown variant '{variant}' for '{family}'. "
            f"Available: {available}"
        )

    target = variants[variant]
    if _is_local_target(target):
        local_path = Path(target).expanduser()
        if not local_path.exists():
            raise ValueError(f"Local model alias '{name}' points to a missing path: {local_path}")
        return str(local_path.resolve())

    return target


def copy_alias(source: str, destination: str) -> str:
    """Create a custom alias for an existing model.

    Returns the repo ID that the new alias points to.
    """
    repo_id = resolve(source)

    dest = destination.strip().lower()
    if not dest:
        raise ValueError("Destination alias cannot be empty.")
    if "/" in dest:
        raise ValueError("Alias cannot contain '/'. Use a simple name like 'mymodel' or 'mymodel:variant'.")

    if ":" in dest:
        family, variant = dest.split(":", 1)
    else:
        family, variant = dest, "default"

    user_registry = _load_user_registry()
    merged = _get_merged_registry()

    if family in merged:
        existing_target = merged[family].get(variant)
        if existing_target == repo_id:
            return repo_id
        if existing_target is not None:
            raise ValueError(
                f"Alias '{destination}' already exists and points to '{existing_target}'. "
                f"Remove it first with `daydream rm {destination}`."
            )

    if family not in user_registry:
        user_registry[family] = {}
    user_registry[family][variant] = repo_id
    _save_user_registry(user_registry)
    return repo_id


def list_user_aliases() -> list[tuple[str, str]]:
    """Return all user-defined aliases as (alias, repo_id) pairs."""
    user_registry = _load_user_registry()
    results: list[tuple[str, str]] = []
    for family, variants in user_registry.items():
        for variant, repo_id in variants.items():
            if variant == "default":
                results.append((family, repo_id))
            else:
                results.append((f"{family}:{variant}", repo_id))
    return sorted(results)


def remove_alias(alias: str) -> str:
    """Remove a user-defined alias. Returns the repo_id it pointed to."""
    alias = alias.strip().lower()
    if ":" in alias:
        family, variant = alias.split(":", 1)
    else:
        family, variant = alias, "default"

    user_registry = _load_user_registry()
    if family not in user_registry or variant not in user_registry[family]:
        raise ValueError(f"Alias '{alias}' not found in user registry.")

    # Check it's not a builtin
    if family in BUILTIN_REGISTRY and variant in BUILTIN_REGISTRY[family]:
        raise ValueError(f"'{alias}' is a built-in name and cannot be removed.")

    repo_id = user_registry[family][variant]
    del user_registry[family][variant]
    if not user_registry[family]:
        del user_registry[family]
    _save_user_registry(user_registry)
    return repo_id


def reverse_lookup(repo_id: str) -> Optional[str]:
    """Find the short name for a HuggingFace repo ID, if one exists."""
    return _find_short_name_for_target(repo_id)


def reverse_lookup_all(repo_id: str) -> list[str]:
    """Find all short names that point to this repo ID."""
    registry = _get_merged_registry()
    names: list[str] = []
    for family, variants in registry.items():
        for variant, value in variants.items():
            if value != repo_id:
                continue
            short = _format_short_name(family, variant, variants)
            if short:
                names.append(short)
    return names


def list_available() -> list[tuple[str, str, str]]:
    """List all available model names: (short_name, variant, repo_id)."""
    scan_local_models(persist=True)
    registry = _get_merged_registry()
    result = []
    for family in sorted(registry.keys()):
        for variant, repo_id in sorted(registry[family].items()):
            short_name = _format_short_name(family, variant, registry[family])
            if short_name is None:
                continue
            result.append((family, variant, repo_id))
    return result

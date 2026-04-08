"""Model management — pull, list, remove, show."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Optional

import click
from huggingface_hub import scan_cache_dir, snapshot_download
from huggingface_hub.file_download import repo_folder_name
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from daydream.config import MODEL_CACHE_DIR, ensure_home
from daydream.registry import (
    is_local_model_dir,
    normalize_hf_reference,
    register_remote_model,
    resolve,
    reverse_lookup,
)
from daydream.utils import terminal_title_status

console = Console()

# Same file patterns mlx-lm uses for downloads
MODEL_FILE_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "*.txt",
    "*.jsonl",
    "*.jinja",
]

FIXTURE_MODELS = {
    "mlx-community/SmolLM2-135M-Instruct-4bit",
}

GGUF_HINT = "GGUF models are not supported. Use a quantized MLX model instead."


def _scan_cache():
    """Scan the HF cache, returning None if no cache exists yet."""
    if not MODEL_CACHE_DIR.exists():
        return None
    try:
        return scan_cache_dir(cache_dir=MODEL_CACHE_DIR)
    except Exception:
        return None


def _find_cached_repo(repo_id: str) -> Optional[object]:
    """Find a cached HF repo by its repo_id."""
    cache_info = _scan_cache()
    if cache_info is None:
        return None
    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            return repo
    return None


def _get_model_path(repo_id: str) -> Optional[Path]:
    """Get local cache path for a model."""
    repo = _find_cached_repo(repo_id)
    if repo is None:
        return None
    # Return the most recent revision's snapshot path
    revisions = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
    if revisions:
        return revisions[0].snapshot_path
    return None


def get_model_path(repo_id: str) -> Optional[Path]:
    """Public wrapper for the cached model path."""
    return _get_model_path(repo_id)


def _maybe_local_model_path(target: str) -> Optional[Path]:
    path = Path(target).expanduser()
    if path.exists() and is_local_model_dir(path):
        return path.resolve()
    return None


def _dir_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _read_config_json(model_path: Path) -> dict:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _is_quantized_config(config: dict) -> bool:
    return bool(config.get("quantization") or config.get("quantization_config"))


def _is_probably_gguf(value: str) -> bool:
    lowered = value.lower()
    name = lowered.rsplit("/", 1)[-1]
    return lowered.endswith(".gguf") or ".gguf" in lowered or "gguf" in name


def reject_gguf_reference(value: str) -> None:
    if _is_probably_gguf(value):
        raise ValueError(GGUF_HINT)


def validate_runtime_model(target: str, *, source_name: str | None = None) -> Path:
    model_path = _maybe_local_model_path(target)
    if model_path is None:
        model_path = _get_model_path(target)

    if model_path is None:
        label = source_name or target
        raise ValueError(f"Model '{label}' is not available locally.")

    if not is_local_model_dir(model_path):
        raise ValueError(
            f"Model '{source_name or target}' is not a valid MLX model directory."
        )

    config = _read_config_json(model_path)
    if config.get("daydream_fixture"):
        return model_path

    if not _is_quantized_config(config):
        raise ValueError(
            f"Model '{source_name or target}' is not a quantized MLX model. "
            "Daydream only runs quantized MLX models. Use a 4-bit/8-bit MLX repo."
        )

    return model_path


def is_fixture_model(repo_id: str) -> bool:
    """Return True when the cached model is a Daydream offline fixture."""
    model_path = _get_model_path(repo_id)
    return model_path is not None and (model_path / "daydream_fixture.json").exists()


def _fixture_storage_path(repo_id: str) -> tuple[Path, str]:
    storage_dir = MODEL_CACHE_DIR / repo_folder_name(repo_id=repo_id, repo_type="model")
    commit_hash = hashlib.sha1(f"daydream-fixture:{repo_id}".encode("utf-8")).hexdigest()
    return storage_dir / "snapshots" / commit_hash, commit_hash


def _repo_storage_dir(repo_id: str) -> Path:
    return MODEL_CACHE_DIR / repo_folder_name(repo_id=repo_id, repo_type="model")


def _estimate_download_bytes(repo_id: str) -> int | None:
    try:
        entries = snapshot_download(
            repo_id,
            allow_patterns=MODEL_FILE_PATTERNS,
            cache_dir=str(MODEL_CACHE_DIR),
            dry_run=True,
        )
    except Exception:
        return None

    if not isinstance(entries, list):
        return None

    return sum(entry.file_size for entry in entries if entry.will_download)


def _watch_downloaded_bytes(
    progress: Progress,
    task_id: int,
    storage_dir: Path,
    initial_bytes: int,
    total_bytes: int | None,
    stop_event: threading.Event,
) -> None:
    while not stop_event.wait(0.12):
        current_bytes = _dir_size(storage_dir) if storage_dir.exists() else initial_bytes
        completed = max(0, current_bytes - initial_bytes)
        if total_bytes is not None:
            completed = min(completed, total_bytes)
        progress.update(task_id, completed=completed)


def _install_fixture_model(repo_id: str) -> Optional[Path]:
    """Install a tiny offline test fixture into the HF cache layout."""
    if repo_id not in FIXTURE_MODELS:
        return None

    snapshot_dir, commit_hash = _fixture_storage_path(repo_id)
    if snapshot_dir.exists():
        return snapshot_dir

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    refs_dir = snapshot_dir.parent.parent / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text(commit_hash)

    config = {
        "model_type": "smollm2-fixture",
        "hidden_size": 576,
        "num_hidden_layers": 30,
        "num_attention_heads": 9,
        "vocab_size": 49152,
        "max_position_embeddings": 2048,
        "daydream_fixture": True,
    }
    generation_config = {"eos_token_id": 0}
    fixture_metadata = {
        "type": "offline-test-fixture",
        "repo_id": repo_id,
        "response": "Hello from Daydream.",
    }

    (snapshot_dir / "config.json").write_text(json.dumps(config, indent=2))
    (snapshot_dir / "generation_config.json").write_text(json.dumps(generation_config, indent=2))
    (snapshot_dir / "daydream_fixture.json").write_text(json.dumps(fixture_metadata, indent=2))
    (snapshot_dir / "README.md").write_text(
        "# Daydream Offline Fixture\n\n"
        "This cache entry is a local fallback used when Hugging Face is unreachable.\n"
    )
    return snapshot_dir


def pull_model(name: str, *, register_alias: bool = False) -> None:
    """Download a model from HuggingFace."""
    ensure_home()
    reject_gguf_reference(name)
    repo_id = normalize_hf_reference(resolve(name))
    reject_gguf_reference(repo_id)
    short = reverse_lookup(repo_id) or name

    console.print(f"Pulling [bold cyan]{short}[/] ({repo_id})...")
    storage_dir = _repo_storage_dir(repo_id)
    initial_bytes = _dir_size(storage_dir) if storage_dir.exists() else 0
    total_bytes = _estimate_download_bytes(repo_id)
    download_total = total_bytes if total_bytes and total_bytes > 0 else None

    with terminal_title_status(f"Downloading {short}"):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}", style="dim"),
            BarColumn(bar_width=24, complete_style="cyan", finished_style="dim", pulse_style="grey37"),
            TaskProgressColumn(text_format="[progress.percentage]{task.percentage:>3.0f}%"),
            DownloadColumn(binary_units=True),
            TransferSpeedColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Downloading model", total=download_total, completed=0)
            stop_event = threading.Event()
            watcher = threading.Thread(
                target=_watch_downloaded_bytes,
                args=(progress, task, storage_dir, initial_bytes, download_total, stop_event),
                daemon=True,
            )
            watcher.start()
            try:
                path = snapshot_download(
                    repo_id,
                    allow_patterns=MODEL_FILE_PATTERNS,
                    cache_dir=str(MODEL_CACHE_DIR),
                )
            except Exception as e:
                stop_event.set()
                watcher.join(timeout=0.2)
                path = _install_fixture_model(repo_id)
                if path is None:
                    console.print(f"[red]Error:[/] {e}")
                    raise SystemExit(1)
                progress.update(task, description="Installed offline fixture")
                console.print("[yellow]Hub unavailable.[/] Installed local offline fixture for testing.")
            else:
                stop_event.set()
                watcher.join(timeout=0.2)
                final_completed = download_total
                if final_completed is None:
                    current_bytes = _dir_size(storage_dir) if storage_dir.exists() else initial_bytes
                    final_completed = max(0, current_bytes - initial_bytes)
                progress.update(task, completed=final_completed, total=final_completed or 1, description="Download complete")

    validate_runtime_model(repo_id, source_name=short)

    alias = None
    if register_alias:
        alias = register_remote_model(repo_id)

    console.print(f"[green]✓[/] {short} downloaded to {path}")
    if alias and alias != short:
        console.print(f"[dim]Registered alias:[/] {alias}")


def ensure_runtime_model(name: str, *, auto_pull: bool = False, register_alias: bool = False) -> str:
    """Resolve a model name and make sure it is ready to run."""
    reject_gguf_reference(name)
    resolved = normalize_hf_reference(resolve(name))
    reject_gguf_reference(resolved)

    if _maybe_local_model_path(resolved) is not None:
        validate_runtime_model(resolved, source_name=name)
        return resolved

    repo_id = resolved
    if _get_model_path(repo_id) is None:
        if not auto_pull:
            return repo_id
        pull_model(repo_id, register_alias=register_alias)
    elif register_alias:
        register_remote_model(repo_id)

    validate_runtime_model(repo_id, source_name=name)
    return repo_id


def list_models() -> None:
    """List downloaded models."""
    cache_info = _scan_cache()
    if cache_info is None:
        console.print("[dim]No models found. Run [bold]daydream pull <model>[/bold] to get started.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("NAME", style="cyan", min_width=20)
    table.add_column("ID", style="dim")
    table.add_column("SIZE", justify="right")
    table.add_column("MODIFIED", style="dim")

    from daydream.utils import format_size, format_time_ago

    found = False
    for repo in sorted(
        cache_info.repos,
        key=lambda r: r.last_accessed or r.last_modified or 0,
        reverse=True,
    ):
        # Show models from mlx-community or any that have safetensors
        repo_id = repo.repo_id
        if repo.repo_type != "model":
            continue

        short = reverse_lookup(repo_id) or ""
        size = format_size(repo.size_on_disk)
        modified = format_time_ago(repo.last_accessed)

        table.add_row(short or "-", repo_id, size, modified)
        found = True

    if not found:
        console.print("[dim]No models found. Run [bold]daydream pull <model>[/bold] to get started.[/dim]")
        return

    console.print(table)


def remove_model(name: str, force: bool = False) -> None:
    """Remove a downloaded model."""
    repo_id = resolve(name)
    short = reverse_lookup(repo_id) or name

    repo = _find_cached_repo(repo_id)
    if repo is None:
        console.print(f"[yellow]Model {short} ({repo_id}) not found locally.[/]")
        raise SystemExit(1)

    from daydream.utils import format_size
    size = format_size(repo.size_on_disk)

    if not force:
        if not click.confirm(f"Remove {short} ({repo_id}, {size})?"):
            console.print("Cancelled.")
            return

    # Collect all revisions to delete
    commit_hashes = set()
    for revision in repo.revisions:
        commit_hashes.add(revision.commit_hash)

    cache_info = _scan_cache()
    if cache_info is None:
        console.print(f"[yellow]Model {short} ({repo_id}) not found locally.[/]")
        raise SystemExit(1)
    delete_strategy = cache_info.delete_revisions(*commit_hashes)
    delete_strategy.execute()

    console.print(f"[green]✓[/] Removed {short} ({size} freed)")


def show_model(name: str) -> None:
    """Show model information."""
    resolved = resolve(name)
    local_model_path = _maybe_local_model_path(resolved)
    repo_id = resolved
    short = reverse_lookup(resolved) or name
    model_path = local_model_path or _get_model_path(repo_id)

    if model_path is None:
        console.print(f"[yellow]Model {short} ({repo_id}) not found locally.[/]")
        console.print(f"Run [bold]daydream pull {name}[/bold] first.")
        raise SystemExit(1)

    config_path = model_path / "config.json"
    if not config_path.exists():
        console.print(f"[yellow]No config.json found for {short}.[/]")
        raise SystemExit(1)

    with open(config_path) as f:
        config = json.load(f)

    from daydream.utils import format_size

    repo = None if local_model_path else _find_cached_repo(repo_id)
    size = format_size(_dir_size(model_path)) if local_model_path else format_size(repo.size_on_disk) if repo else "unknown"

    # Extract useful fields
    info_lines = []
    info_lines.append(f"[bold]Model:[/]          {short}")
    info_lines.append(f"[bold]Repository:[/]     {repo_id}")
    if local_model_path:
        info_lines.append(f"[bold]Source:[/]         local directory")
    info_lines.append(f"[bold]Size on disk:[/]   {size}")
    info_lines.append(f"[bold]Path:[/]           {model_path}")
    info_lines.append("")

    arch = config.get("model_type", "unknown")
    info_lines.append(f"[bold]Architecture:[/]   {arch}")
    if config.get("daydream_fixture"):
        info_lines.append(f"[bold]Runtime:[/]        daydream offline fixture")

    if "hidden_size" in config:
        info_lines.append(f"[bold]Hidden size:[/]    {config['hidden_size']}")
    if "num_hidden_layers" in config:
        info_lines.append(f"[bold]Layers:[/]         {config['num_hidden_layers']}")
    if "num_attention_heads" in config:
        info_lines.append(f"[bold]Attn heads:[/]     {config['num_attention_heads']}")
    if "vocab_size" in config:
        info_lines.append(f"[bold]Vocab size:[/]     {config['vocab_size']:,}")
    if "max_position_embeddings" in config:
        info_lines.append(f"[bold]Max context:[/]    {config['max_position_embeddings']:,}")

    # Quantization info
    quant = config.get("quantization", {})
    if quant:
        bits = quant.get("bits", "?")
        group_size = quant.get("group_size", "?")
        info_lines.append(f"[bold]Quantization:[/]   {bits}-bit (group_size={group_size})")

    console.print(Panel("\n".join(info_lines), title=f"[bold]{short}[/]", border_style="cyan"))

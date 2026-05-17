"""Thin MTP install — reuse the user's existing Qwen3.6 trunk weights.

`Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` is ~16.4 GB on disk but
only **338 MB of that is genuinely new**: the rest is a byte-identical
copy of `mlx-community/Qwen3.6-27B-4bit` that Youssofal `clonefile()`'d
into their repo for self-containment (see MTPLX_PUBLISH_MANIFEST.json).

This module installs MTP support by:
    1. Downloading ONLY the unique files (`mtp.safetensors` + the
       runtime contract) — ~338 MB.
    2. Symlinking the trunk + tokenizer from the user's existing
       `mlx-community/Qwen3.6-27B-4bit` cache.
    3. Assembling everything in `~/.daydream/mtp/<trunk-slug>/` so it's
       trivially uninstallable with `daydream mtp uninstall` or by
       removing one directory.

If the user later removes the standard Qwen3.6 trunk via `daydream rm`,
the MTP install becomes a pile of broken symlinks — `find_thin_install`
detects this and reports `broken_trunk=True` so the CLI can offer to
re-link or clean up.

The MTPLX runtime (`mtplx.load(path)`) accepts the synthetic directory
because mlx-lm's loader walks `model.safetensors.index.json` and reads
each file by name; whether those names are real files or symlinks is
transparent to it.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from daydream.config import DAYDREAM_HOME, ensure_home

# Files we genuinely need to fetch from the MTPLX repo. Sizes are from
# MTPLX_PUBLISH_MANIFEST.json so we can show an honest progress estimate
# without first calling the Hub.
#
# CRITICAL: config.json is in DOWNLOAD_FILES (not SYMLINK) because the
# MTPLX repo augments the standard mlx-community config with an
# `mtplx_mtp_quantization` block declaring the prequantized MTP head's
# bits + group_size. Symlinking the trunk's config.json would let
# MTPLX fall back to its default `group_size=64`, but the actual
# mtp.safetensors is `group_size=32` — the shape check then fails
# with: "scales.shape() == (12288,160) ... group_size=64 and bits=4".
DOWNLOAD_FILES: tuple[tuple[str, int], ...] = (
    ("mtp.safetensors", 337_565_787),
    ("mtplx_runtime.json", 2_007),
    ("config.json", 5_541),
)
THIN_INSTALL_DOWNLOAD_BYTES = sum(size for _, size in DOWNLOAD_FILES)

# Files we symlink from the trunk. Order matters for the loader: it
# reads `model.safetensors.index.json` first, then resolves the per-
# shard safetensors by name. `config.json` is intentionally NOT here
# — see DOWNLOAD_FILES.
SYMLINK_FILES_REQUIRED: tuple[str, ...] = (
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
)
SYMLINK_FILES_OPTIONAL: tuple[str, ...] = (
    "chat_template.jinja",
    "vocab.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "configuration.json",
)

DEFAULT_SIDECAR_REPO = "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
DEFAULT_BASE_TRUNK = "mlx-community/Qwen3.6-27B-4bit"


@dataclass(frozen=True)
class ThinInstall:
    install_dir: Path
    base_trunk: str
    sidecar_repo: str
    bytes_downloaded: int
    bytes_symlinked: int
    broken_trunk: bool


def _slug(value: str) -> str:
    """Filesystem-safe slug for a HuggingFace repo id."""
    return value.replace("/", "--").replace(" ", "-").lower()


def _install_root() -> Path:
    """`~/.daydream/mtp/` — created on demand."""
    ensure_home()
    root = DAYDREAM_HOME / "mtp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _trunk_local_path(base_trunk: str) -> Optional[Path]:
    """Locate the user's existing trunk checkpoint in the HF cache."""
    # Lazy import — avoids dragging in daydream.models (which itself
    # pulls click + rich) when we only need this helper.
    from daydream.models import get_model_path

    path = get_model_path(base_trunk)
    return path if path is not None and path.is_dir() else None


def _file_total_size(paths: list[Path]) -> int:
    total = 0
    for p in paths:
        try:
            # Follow symlinks so we report the trunk's real size, not 0.
            total += p.stat().st_size
        except OSError:
            continue
    return total


def find_thin_install(base_trunk: str = DEFAULT_BASE_TRUNK) -> Optional[ThinInstall]:
    """Return an existing ThinInstall if one is set up, else None.

    Reports `broken_trunk=True` when the install dir exists but one or
    more symlinks point at a missing trunk (e.g. the user ran
    `daydream rm` on the trunk).
    """
    install_dir = _install_root() / _slug(base_trunk)
    if not install_dir.is_dir():
        return None
    runtime_json = install_dir / "mtplx_runtime.json"
    if not runtime_json.is_file():
        return None  # half-finished install — treat as absent

    # Detect broken symlinks (trunk was removed since install).
    broken_trunk = False
    symlinked_paths: list[Path] = []
    for name in SYMLINK_FILES_REQUIRED:
        target = install_dir / name
        if not target.exists():
            broken_trunk = True
        else:
            symlinked_paths.append(target)

    downloaded_paths = [install_dir / name for name, _ in DOWNLOAD_FILES]
    return ThinInstall(
        install_dir=install_dir,
        base_trunk=base_trunk,
        sidecar_repo=DEFAULT_SIDECAR_REPO,
        bytes_downloaded=_file_total_size([p for p in downloaded_paths if p.is_file() and not p.is_symlink()]),
        bytes_symlinked=_file_total_size(symlinked_paths),
        broken_trunk=broken_trunk,
    )


def thin_install(
    *,
    base_trunk: str = DEFAULT_BASE_TRUNK,
    sidecar_repo: str = DEFAULT_SIDECAR_REPO,
    progress: bool = True,
    full: bool = False,
) -> ThinInstall:
    """Install MTP support.

    `full=False` (default): symlink the trunk from the user's existing
    `mlx-community` cache + download only mtp.safetensors (~338 MB).
    Works when the user's local trunk has the same quantization params
    as the trunk MTPLX was packaged against.

    `full=True`: download the FULL MTPLX-packaged repo (~16 GB),
    including the bundled trunk safetensors. Use this when the thin
    install fails with a shape/quantization mismatch — the bundled
    trunk is guaranteed to match mtp.safetensors.

    Raises:
        FileNotFoundError: trunk is not cached locally (thin mode only).
    """
    install_dir = _install_root() / _slug(base_trunk)
    install_dir.mkdir(parents=True, exist_ok=True)

    bytes_symlinked = 0
    bytes_downloaded = 0
    from huggingface_hub import hf_hub_download

    if full:
        # ── Full install: pull every file from the MTPLX repo. ─────
        # No symlinks; the bundled trunk shards ARE the trunk MTPLX
        # was packaged against — quantization params match by
        # construction.
        files_to_pull = SYMLINK_FILES_REQUIRED + tuple(n for n, _ in DOWNLOAD_FILES) + SYMLINK_FILES_OPTIONAL
        for filename in files_to_pull:
            try:
                if progress:
                    from daydream.models import progress_console
                    progress_console.print(f"[dim]Downloading {filename}...[/dim]")
                local_path = hf_hub_download(
                    repo_id=sidecar_repo,
                    filename=filename,
                    local_dir=str(install_dir),
                )
                bytes_downloaded += Path(local_path).stat().st_size
            except Exception:
                # OPTIONAL files may legitimately be absent (e.g.
                # video_preprocessor_config.json on text-only repos).
                if filename in SYMLINK_FILES_REQUIRED or filename in {n for n, _ in DOWNLOAD_FILES}:
                    raise

        return ThinInstall(
            install_dir=install_dir,
            base_trunk=base_trunk,
            sidecar_repo=sidecar_repo,
            bytes_downloaded=bytes_downloaded,
            bytes_symlinked=0,
            broken_trunk=False,
        )

    # ── Thin install (default): symlink trunk + download MTP head. ─
    trunk_dir = _trunk_local_path(base_trunk)
    if trunk_dir is None:
        raise FileNotFoundError(
            f"Base trunk `{base_trunk}` is not cached locally. "
            f"Run `daydream pull {base_trunk}` first, or use "
            "`daydream mtp install --full` to pull the bundled trunk."
        )

    for name in SYMLINK_FILES_REQUIRED + SYMLINK_FILES_OPTIONAL:
        src = trunk_dir / name
        if not src.exists():
            if name in SYMLINK_FILES_REQUIRED:
                raise FileNotFoundError(
                    f"Trunk `{base_trunk}` is missing `{name}` — "
                    "the local checkpoint may be incomplete. Re-run "
                    f"`daydream pull {base_trunk}` to repair."
                )
            continue
        dst = install_dir / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
        try:
            bytes_symlinked += src.stat().st_size
        except OSError:
            pass

    for filename, _expected_size in DOWNLOAD_FILES:
        if progress:
            from daydream.models import progress_console
            progress_console.print(f"[dim]Downloading {filename} from {sidecar_repo}...[/dim]")
        local_path = hf_hub_download(
            repo_id=sidecar_repo,
            filename=filename,
            local_dir=str(install_dir),
        )
        try:
            bytes_downloaded += Path(local_path).stat().st_size
        except OSError:
            pass

    return ThinInstall(
        install_dir=install_dir,
        base_trunk=base_trunk,
        sidecar_repo=sidecar_repo,
        bytes_downloaded=bytes_downloaded,
        bytes_symlinked=bytes_symlinked,
        broken_trunk=False,
    )


def thin_uninstall(*, base_trunk: str = DEFAULT_BASE_TRUNK) -> int:
    """Remove the thin-install dir. Returns bytes freed (the unique
    download portion; symlinks reclaim 0 bytes because the trunk stays).
    """
    install_dir = _install_root() / _slug(base_trunk)
    if not install_dir.is_dir():
        return 0
    bytes_freed = 0
    for name, _ in DOWNLOAD_FILES:
        target = install_dir / name
        if target.is_file() and not target.is_symlink():
            try:
                bytes_freed += target.stat().st_size
            except OSError:
                pass
    shutil.rmtree(install_dir, ignore_errors=True)
    return bytes_freed

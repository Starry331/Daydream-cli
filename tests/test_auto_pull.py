from __future__ import annotations

import importlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rich.console import Console


def reload_module(name: str):
    return importlib.reload(importlib.import_module(name))


def write_remote_cache_model(path: Path, *, quantized: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config = {"model_type": "qwen"}
    if quantized:
        config["quantization"] = {"bits": 4, "group_size": 64}
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "model.safetensors").write_text("weights", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


class AutoPullTests(unittest.TestCase):
    def test_hf_reference_is_normalized_and_registered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            home.mkdir(parents=True, exist_ok=True)

            with mock.patch.dict(os.environ, {"DAYDREAM_HOME": str(home)}, clear=False):
                reload_module("daydream.config")
                registry = reload_module("daydream.registry")

                self.assertEqual(
                    registry.normalize_hf_reference("hf.co/acme/Qwen3.5-27B-MLX-4bit"),
                    "acme/Qwen3.5-27B-MLX-4bit",
                )
                alias = registry.register_remote_model("hf.co/acme/Qwen3.5-27B-MLX-4bit")
                self.assertEqual(alias, "qwen3.5-27b")
                self.assertEqual(
                    registry.resolve("qwen3.5-27b"),
                    "acme/Qwen3.5-27B-MLX-4bit",
                )

    def test_ensure_runtime_model_auto_pulls_quantized_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            cache = Path(tmpdir) / "cache"
            output = io.StringIO()
            downloaded = cache / "models--acme--Qwen3.5-27B-MLX-4bit" / "snapshots" / "abc123"

            with mock.patch.dict(
                os.environ,
                {
                    "DAYDREAM_HOME": str(home),
                    "DAYDREAM_CACHE_DIR": str(cache),
                },
                clear=False,
            ):
                reload_module("daydream.config")
                reload_module("daydream.registry")
                models = reload_module("daydream.models")
                models.console = Console(file=output, force_terminal=False, color_system=None)

                def fake_snapshot_download_with_dry_run(repo_id, allow_patterns=None, cache_dir=None, dry_run=False, **kwargs):
                    self.assertEqual(repo_id, "acme/Qwen3.5-27B-MLX-4bit")
                    if dry_run:
                        class _Entry:
                            file_size = 1024
                            will_download = True
                        return [_Entry()]
                    write_remote_cache_model(downloaded, quantized=True)
                    return str(downloaded)

                with mock.patch.object(models, "snapshot_download", side_effect=fake_snapshot_download_with_dry_run):
                    resolved = models.ensure_runtime_model(
                        "hf.co/acme/Qwen3.5-27B-MLX-4bit",
                        auto_pull=True,
                        register_alias=True,
                    )

                self.assertEqual(resolved, "acme/Qwen3.5-27B-MLX-4bit")
                self.assertEqual(models.reverse_lookup(resolved), "qwen3.5-27b")

    def test_ensure_runtime_model_rejects_unquantized_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            cache = Path(tmpdir) / "cache"
            downloaded = cache / "models--acme--Plain-MLX" / "snapshots" / "abc123"

            with mock.patch.dict(
                os.environ,
                {
                    "DAYDREAM_HOME": str(home),
                    "DAYDREAM_CACHE_DIR": str(cache),
                },
                clear=False,
            ):
                reload_module("daydream.config")
                reload_module("daydream.registry")
                models = reload_module("daydream.models")

                def fake_snapshot_download(repo_id, allow_patterns=None, cache_dir=None, dry_run=False, **kwargs):
                    if dry_run:
                        class _Entry:
                            file_size = 1024
                            will_download = True
                        return [_Entry()]
                    write_remote_cache_model(downloaded, quantized=False)
                    return str(downloaded)

                with mock.patch.object(models, "snapshot_download", side_effect=fake_snapshot_download):
                    with self.assertRaisesRegex(ValueError, "not a quantized MLX model"):
                        models.ensure_runtime_model(
                            "hf.co/acme/Plain-MLX",
                            auto_pull=True,
                            register_alias=True,
                        )

    def test_ensure_runtime_model_rejects_gguf_early(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            cache = Path(tmpdir) / "cache"

            with mock.patch.dict(
                os.environ,
                {
                    "DAYDREAM_HOME": str(home),
                    "DAYDREAM_CACHE_DIR": str(cache),
                },
                clear=False,
            ):
                reload_module("daydream.config")
                reload_module("daydream.registry")
                models = reload_module("daydream.models")

                with self.assertRaisesRegex(ValueError, "GGUF models are not supported"):
                    models.ensure_runtime_model("hf.co/bartowski/Qwen3.5-14B-GGUF", auto_pull=True)


if __name__ == "__main__":
    unittest.main()

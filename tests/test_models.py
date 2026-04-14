from __future__ import annotations

import importlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rich.console import Console


def reload_module(name: str):
    return importlib.reload(importlib.import_module(name))


class ModelTests(unittest.TestCase):
    def test_pull_installs_fixture_when_hub_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            cache = Path(tmpdir) / "cache"
            output = io.StringIO()

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

                def fake_snapshot_download(*args, dry_run=False, **kwargs):
                    if dry_run:
                        class _Entry:
                            file_size = 1024
                            will_download = True
                        return [_Entry()]
                    raise RuntimeError("offline")

                with mock.patch.object(models, "snapshot_download", side_effect=fake_snapshot_download):
                    models.pull_model("smollm2:135m")

                repo_id = "mlx-community/SmolLM2-135M-Instruct-4bit"
                model_path = models.get_model_path(repo_id)
                self.assertIsNotNone(model_path)
                self.assertTrue(model_path.exists())
                self.assertTrue(models.is_fixture_model(repo_id))

                output.truncate(0)
                output.seek(0)
                models.list_models()
                rendered = output.getvalue()
                self.assertIn("smollm2:135m", rendered)
                self.assertIn("mlx-community/SmolLM2-135M", rendered)

    def test_pull_falls_back_to_single_line_status_when_progress_console_is_not_interactive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            cache = Path(tmpdir) / "cache"
            output = io.StringIO()
            downloaded = cache / "models--acme--Qwen3.5-9B-MLX-4bit" / "snapshots" / "abc123"

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
                models.progress_console = Console(file=io.StringIO(), force_terminal=False, color_system=None)

                def fake_snapshot_download(*args, dry_run=False, **kwargs):
                    if dry_run:
                        class _Entry:
                            file_size = 1024
                            will_download = True
                        return [_Entry()]
                    from tests.test_auto_pull import write_remote_cache_model
                    write_remote_cache_model(downloaded, quantized=True)
                    return str(downloaded)

                with mock.patch.object(models, "snapshot_download", side_effect=fake_snapshot_download), \
                    mock.patch.object(models, "Progress") as progress_cls:
                    models.pull_model("hf.co/acme/Qwen3.5-9B-MLX-4bit")

                progress_cls.assert_not_called()
                rendered = output.getvalue()
                self.assertIn("Downloading hf.co/acme/Qwen3.5-9B-MLX-4bit", rendered)


if __name__ == "__main__":
    unittest.main()

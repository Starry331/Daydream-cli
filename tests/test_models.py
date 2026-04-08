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


if __name__ == "__main__":
    unittest.main()

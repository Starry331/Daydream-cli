from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def reload_module(name: str):
    return importlib.reload(importlib.import_module(name))


def write_fake_local_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type":"qwen"}', encoding="utf-8")
    (path / "model.safetensors").write_text("weights", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


class LocalRegistryTests(unittest.TestCase):
    def test_direct_local_path_auto_registers_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            local_model = Path(tmpdir) / "Qwen3.5-9B-MLX-4bit"
            home.mkdir(parents=True, exist_ok=True)
            write_fake_local_model(local_model)

            with mock.patch.dict(os.environ, {"DAYDREAM_HOME": str(home)}, clear=False):
                reload_module("daydream.config")
                registry = reload_module("daydream.registry")

                resolved = registry.resolve(str(local_model))
                self.assertEqual(resolved, str(local_model.resolve()))
                self.assertEqual(registry.resolve("qwen3.5-9b"), str(local_model.resolve()))
                self.assertEqual(registry.reverse_lookup(str(local_model.resolve())), "qwen3.5-9b")

    def test_scan_local_model_roots_registers_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            root = Path(tmpdir) / "models"
            local_model = root / "nested" / "My-Coder-MLX-4bit"
            home.mkdir(parents=True, exist_ok=True)
            write_fake_local_model(local_model)

            with mock.patch.dict(
                os.environ,
                {
                    "DAYDREAM_HOME": str(home),
                    "DAYDREAM_MODELS_DIRS": str(root),
                },
                clear=False,
            ):
                reload_module("daydream.config")
                registry = reload_module("daydream.registry")

                discovered = registry.scan_local_models(persist=True)
                self.assertIn(("my-coder", str(local_model.resolve())), discovered)
                self.assertEqual(registry.resolve("my-coder"), str(local_model.resolve()))


if __name__ == "__main__":
    unittest.main()

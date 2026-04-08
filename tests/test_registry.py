from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def reload_module(name: str):
    return importlib.reload(importlib.import_module(name))


class RegistryTests(unittest.TestCase):
    def test_user_registry_overrides_and_adds_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            home.mkdir(parents=True, exist_ok=True)
            (home / "registry.yaml").write_text(
                "qwen3:\n"
                "  4b: acme/custom-qwen3-4b\n"
                "custom:\n"
                "  default: acme/custom-default\n"
                "  mini: acme/custom-mini\n",
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"DAYDREAM_HOME": str(home)}, clear=False):
                reload_module("daydream.config")
                registry = reload_module("daydream.registry")

                self.assertEqual(registry.resolve("qwen3:4b"), "acme/custom-qwen3-4b")
                self.assertEqual(registry.resolve("custom"), "acme/custom-default")
                self.assertEqual(registry.resolve("custom:mini"), "acme/custom-mini")
                self.assertEqual(registry.resolve("mlx-community/foo"), "mlx-community/foo")


if __name__ == "__main__":
    unittest.main()

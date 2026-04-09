from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def reload_module(name: str):
    return importlib.reload(importlib.import_module(name))


class ConfigTests(unittest.TestCase):
    def test_config_uses_env_overrides_and_yaml_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "daydream-home"
            cache = Path(tmpdir) / "hf-cache"
            home.mkdir(parents=True, exist_ok=True)
            (home / "config.yaml").write_text(
                "model: qwen3:4b\n"
                "run:\n"
                "  temp: 0.2\n"
                "  top_p: 0.8\n"
                "  max_tokens: 1024\n"
                "serve:\n"
                "  host: 0.0.0.0\n"
                "  port: 8080\n",
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ,
                {
                    "DAYDREAM_HOME": str(home),
                    "DAYDREAM_CACHE_DIR": str(cache),
                },
                clear=False,
            ):
                config = reload_module("daydream.config")
                self.assertEqual(config.DAYDREAM_HOME, home)
                self.assertEqual(config.MODEL_CACHE_DIR, cache)
                self.assertEqual(config.get_default_model(), "qwen3:4b")
                self.assertEqual(config.get_default_temp(), 0.2)
                self.assertEqual(config.get_default_top_p(), 0.8)
                self.assertEqual(config.get_default_max_tokens(), 1024)
                self.assertEqual(config.get_default_host(), "0.0.0.0")
                self.assertEqual(config.get_default_port(), 8080)
                self.assertEqual(config.get_default_cli_page_mode(), "loose")

    def test_ensure_home_creates_chat_and_memory_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "daydream-home"

            with mock.patch.dict(
                os.environ,
                {"DAYDREAM_HOME": str(home)},
                clear=False,
            ):
                config = reload_module("daydream.config")
                config.ensure_home()

                self.assertTrue(config.DAYDREAM_HOME.exists())
                self.assertTrue(config.CHATS_DIR.exists())
                self.assertTrue(config.MEMORIES_DIR.exists())

    def test_cli_page_mode_defaults_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "daydream-home"
            home.mkdir(parents=True, exist_ok=True)

            with mock.patch.dict(
                os.environ,
                {"DAYDREAM_HOME": str(home)},
                clear=False,
            ):
                config = reload_module("daydream.config")
                self.assertEqual(config.get_default_cli_page_mode(), "loose")

                persisted = config.set_default_cli_page_mode("tight")
                self.assertEqual(persisted, "tight")

                config = reload_module("daydream.config")
                self.assertEqual(config.get_default_cli_page_mode(), "tight")

                (home / "config.yaml").write_text(
                    "chat:\n  cli_page_mode: invalid\n",
                    encoding="utf-8",
                )
                config = reload_module("daydream.config")
                self.assertEqual(config.get_default_cli_page_mode(), "loose")


if __name__ == "__main__":
    unittest.main()

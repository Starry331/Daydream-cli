from __future__ import annotations

import importlib
import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock


def reload_module(name: str):
    return importlib.reload(importlib.import_module(name))


class ProfileTests(unittest.TestCase):
    def test_parse_daydreamfile_supports_system_and_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            daydreamfile = Path(tmpdir) / "Daydreamfile"
            daydreamfile.write_text(
                textwrap.dedent(
                    '''\
                    FROM qwen3:8b
                    SYSTEM """
                    You are a terse coding assistant.
                    Keep answers short.
                    """
                    PARAMETER temperature 0.2
                    PARAMETER top_p 0.95
                    PARAMETER max_tokens 8192
                    PARAMETER effort long
                    '''
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            profiles = reload_module("daydream.profiles")
            profile = profiles.parse_daydreamfile(daydreamfile, name="coder")

        self.assertEqual(profile.name, "coder")
        self.assertEqual(profile.from_model, "qwen3:8b")
        self.assertIn("terse coding assistant", profile.system or "")
        self.assertEqual(profile.parameters["temperature"], 0.2)
        self.assertEqual(profile.parameters["top_p"], 0.95)
        self.assertEqual(profile.parameters["max_tokens"], 8192)
        self.assertEqual(profile.parameters["effort"], "long")

    def test_create_and_load_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            home.mkdir(parents=True, exist_ok=True)
            daydreamfile = Path(tmpdir) / "Daydreamfile"
            daydreamfile.write_text(
                "FROM mlx-community/Qwen3-8B-4bit\nPARAMETER temperature 0.4\n",
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"DAYDREAM_HOME": str(home)}, clear=False):
                reload_module("daydream.config")
                profiles = reload_module("daydream.profiles")
                created = profiles.create_profile("mycoder", file_path=daydreamfile)
                loaded = profiles.get_profile("mycoder")

            self.assertEqual(created.name, "mycoder")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.from_model, "mlx-community/Qwen3-8B-4bit")
            self.assertEqual(loaded.parameters["temperature"], 0.4)


if __name__ == "__main__":
    unittest.main()

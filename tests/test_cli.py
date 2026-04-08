from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from click.testing import CliRunner

from daydream.cli import _coalesce_model_reference, cli


class CliTests(unittest.TestCase):
    def test_coalesce_model_reference_supports_space_separated_short_name(self) -> None:
        self.assertEqual(_coalesce_model_reference(None, ("qwen3.5", "9b")), "qwen3.5:9b")
        self.assertEqual(_coalesce_model_reference(None, ("qwen2.5", "coder", "7b")), "qwen2.5-coder:7b")

    def test_coalesce_model_reference_rejects_duplicate_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "Use either a positional model or --model"):
            _coalesce_model_reference("qwen3:8b", ("qwen3", "8b"))

    def test_serve_accepts_positional_model_reference(self) -> None:
        runner = CliRunner()
        with mock.patch("daydream.server.start_server") as start_server:
            result = runner.invoke(cli, ["serve", "qwen3.5", "9b"])

        self.assertEqual(result.exit_code, 0, result.output)
        start_server.assert_called_once_with(
            model="qwen3.5:9b",
            host="127.0.0.1",
            port=11434,
            detach=False,
        )

    def test_serve_accepts_legacy_model_option(self) -> None:
        runner = CliRunner()
        with mock.patch("daydream.server.start_server") as start_server:
            result = runner.invoke(cli, ["serve", "--model", "qwen3:8b", "--background"])

        self.assertEqual(result.exit_code, 0, result.output)
        start_server.assert_called_once_with(
            model="qwen3:8b",
            host="127.0.0.1",
            port=11434,
            detach=True,
        )

    def test_create_command_saves_profile(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            home.mkdir(parents=True, exist_ok=True)
            daydreamfile = Path(tmpdir) / "Daydreamfile"
            daydreamfile.write_text("FROM qwen3:8b\nPARAMETER effort long\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {"DAYDREAM_HOME": str(home)}, clear=False):
                import importlib
                importlib.reload(__import__("daydream.config", fromlist=["*"]))
                importlib.reload(__import__("daydream.profiles", fromlist=["*"]))
                result = runner.invoke(cli, ["create", "mycoder", "-f", str(daydreamfile)])

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue((home / "profiles.yaml").exists())

    def test_run_uses_profile_defaults_when_flags_not_explicit(self) -> None:
        runner = CliRunner()
        with mock.patch("daydream.cli._resolve_profile_reference") as resolve_profile, \
            mock.patch("daydream.chat.run_oneshot") as run_oneshot:
            profile = mock.Mock()
            profile.name = "mycoder"
            profile.system = "You are terse."
            profile.parameters = {
                "temperature": 0.2,
                "top_p": 0.8,
                "max_tokens": 9000,
                "effort": "long",
            }
            resolve_profile.return_value = ("qwen3:8b", profile)
            result = runner.invoke(cli, ["run", "mycoder", "hello"])

        self.assertEqual(result.exit_code, 0, result.output)
        run_oneshot.assert_called_once_with(
            "qwen3:8b",
            prompt="hello",
            temp=0.2,
            top_p=0.8,
            max_tokens=9000,
            system="You are terse.",
            verbose=False,
            initial_effort="long",
            display_name="mycoder",
        )


if __name__ == "__main__":
    unittest.main()

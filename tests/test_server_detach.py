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


class ServerDetachTests(unittest.TestCase):
    def test_detach_spawns_background_process_and_writes_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            home.mkdir(parents=True, exist_ok=True)

            with mock.patch.dict(os.environ, {"DAYDREAM_HOME": str(home)}, clear=False):
                reload_module("daydream.config")
                server = reload_module("daydream.server")
                output = io.StringIO()
                server.console = Console(file=output, force_terminal=False, color_system=None)

                process = mock.Mock()
                process.pid = 43210
                process.poll.return_value = None

                with mock.patch.object(server, "ensure_runtime_model", return_value="mlx-community/Foo-4bit"), \
                    mock.patch.object(server, "_is_server_healthy", side_effect=[False, True]), \
                    mock.patch("subprocess.Popen", return_value=process):
                    server.start_server(model="hf.co/mlx-community/Foo-4bit", host="127.0.0.1", port=11434, detach=True)

                state = json.loads((home / "server.json").read_text(encoding="utf-8"))
                self.assertEqual(state["pid"], 43210)
                self.assertEqual(state["model"], "mlx-community/Foo-4bit")
                self.assertEqual(state["port"], 11434)
                rendered = output.getvalue()
                self.assertIn("Server started in background", rendered)

    def test_stop_server_terminates_managed_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            home.mkdir(parents=True, exist_ok=True)
            (home / "server.json").write_text(
                json.dumps(
                    {
                        "pid": 43210,
                        "host": "127.0.0.1",
                        "port": 11434,
                        "model": "mlx-community/Foo-4bit",
                        "log_file": str(home / "server.log"),
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"DAYDREAM_HOME": str(home)}, clear=False):
                reload_module("daydream.config")
                server = reload_module("daydream.server")
                output = io.StringIO()
                server.console = Console(file=output, force_terminal=False, color_system=None)

                with mock.patch.object(server, "_pid_is_running", side_effect=[True, False]), \
                    mock.patch("os.kill") as mock_kill:
                    server.stop_server()

                mock_kill.assert_called_once()
                self.assertFalse((home / "server.json").exists())
                self.assertIn("Stopped server", output.getvalue())

    def test_background_mode_spawns_background_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            home.mkdir(parents=True, exist_ok=True)

            with mock.patch.dict(os.environ, {"DAYDREAM_HOME": str(home)}, clear=False):
                reload_module("daydream.config")
                server = reload_module("daydream.server")

                with mock.patch.object(server, "_spawn_background_server") as spawn, \
                    mock.patch.object(server, "ensure_runtime_model", return_value="mlx-community/Foo-4bit"):
                    server.start_server(model="hf.co/mlx-community/Foo-4bit", detach=True)

                spawn.assert_called_once()

    def test_serve_defaults_to_foreground_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            home.mkdir(parents=True, exist_ok=True)

            with mock.patch.dict(os.environ, {"DAYDREAM_HOME": str(home)}, clear=False):
                reload_module("daydream.config")
                server = reload_module("daydream.server")
                output = io.StringIO()
                server.console = Console(file=output, force_terminal=False, color_system=None)

                with mock.patch.object(server, "_spawn_background_server") as spawn, \
                    mock.patch.object(server, "ensure_runtime_model", return_value="mlx-community/Foo-4bit"), \
                    mock.patch.object(server, "is_fixture_model", return_value=True), \
                    mock.patch.object(server, "_run_fixture_server") as run_fixture:
                    server.start_server(model="hf.co/mlx-community/Foo-4bit")

                spawn.assert_not_called()
                run_fixture.assert_called_once_with("127.0.0.1", 11434, "mlx-community/Foo-4bit")
                self.assertIn("server listening on", output.getvalue())


if __name__ == "__main__":
    unittest.main()

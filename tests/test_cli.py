from __future__ import annotations

import unittest
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()

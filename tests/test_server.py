from __future__ import annotations

import importlib
import io
import json
import unittest
from unittest import mock

from rich.console import Console


def reload_module(name: str):
    return importlib.reload(importlib.import_module(name))


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class ServerTests(unittest.TestCase):
    def test_finalize_runtime_response_text_keeps_reasoning_out_of_content(self) -> None:
        server = reload_module("daydream.server")
        parser = server._ReasoningParser()
        mixed = (
            "Usually, this means I should output the reasoning.\n"
            "Let's check the system instruction again: Keep it short.\n"
            "*Final Answer:* Hello! How can I help you today?"
        )
        visible, _, _ = parser.feed(mixed)
        content, reasoning = server._finalize_runtime_response_text(visible, parser)

        self.assertEqual(content, "Hello! How can I help you today?")
        self.assertIn("Usually, this means", reasoning)
        self.assertNotIn("Usually, this means", content)

    def test_finalize_runtime_response_text_extracts_chinese_answer_label(self) -> None:
        server = reload_module("daydream.server")
        parser = server._ReasoningParser()
        mixed = "分析：用户发送“？”意在确认响应状态。\n回答：我在，有什么可以帮您的吗？"
        visible, _, _ = parser.feed(mixed)
        content, reasoning = server._finalize_runtime_response_text(visible, parser)

        self.assertEqual(content, "我在，有什么可以帮您的吗？")
        self.assertIn("分析：用户发送", reasoning)
        self.assertNotIn("分析：用户发送", content)

    def test_show_status_checks_health_and_lists_models(self) -> None:
        server = reload_module("daydream.server")
        output = io.StringIO()
        server.console = Console(file=output, force_terminal=False, color_system=None)

        calls: list[str] = []

        class _FakeOpener:
            def open(self, url, timeout=0):
                calls.append(url)
                if url.endswith("/health"):
                    return _FakeResponse({"status": "ok"})
                if url.endswith("/v1/models"):
                    return _FakeResponse(
                        {
                            "data": [
                                {
                                    "id": "mlx-community/SmolLM2-135M-Instruct-4bit",
                                }
                            ]
                        }
                    )
                raise AssertionError(f"Unexpected URL: {url}")

        with mock.patch("urllib.request.build_opener", return_value=_FakeOpener()):
            server.show_status()

        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0].endswith("/health"))
        self.assertTrue(calls[1].endswith("/v1/models"))
        rendered = output.getvalue()
        self.assertIn("Server running", rendered)
        self.assertIn("smollm2:135m", rendered)
        self.assertIn("OpenAI:   http://127.0.0.1:11434/v1", rendered)
        self.assertIn("API key: any non-empty value", rendered)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


def reload_module(name: str):
    return importlib.reload(importlib.import_module(name))


class StorageTests(unittest.TestCase):
    def test_save_and_load_session_and_memories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "daydream-home"
            with mock.patch.dict(os.environ, {"DAYDREAM_HOME": str(home)}, clear=False):
                reload_module("daydream.config")
                storage = reload_module("daydream.storage")

                session = storage.ChatSession(
                    session_id="abc123",
                    model="qwen3:8b",
                    title="",
                    created_at=time.time(),
                    updated_at=time.time(),
                    messages=[
                        storage.ChatMessage(role="user", content="Remember I like concise answers.", timestamp=time.time()),
                        storage.ChatMessage(role="assistant", content="Noted.", timestamp=time.time(), reasoning="hidden"),
                    ],
                )
                storage.save_session(session)

                loaded = storage.load_session("abc123")
                self.assertIsNotNone(loaded)
                assert loaded is not None
                self.assertEqual(loaded.title, "Remember I like concise answers.")
                self.assertEqual(len(loaded.messages), 2)
                self.assertEqual(loaded.messages[1].reasoning, "hidden")

                memories = [
                    storage.Memory(
                        content="User likes concise answers.",
                        category="preference",
                        importance=0.9,
                        source_phase="reming",
                        created_at=time.time(),
                        session_id="abc123",
                    )
                ]
                storage.save_memories("abc123", memories)
                loaded_memories = storage.load_memories("abc123")
                self.assertEqual(len(loaded_memories), 1)
                self.assertEqual(loaded_memories[0].content, "User likes concise answers.")
                self.assertEqual(len(storage.load_all_memories()), 1)

    def test_list_sessions_sorts_by_updated_at_descending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "daydream-home"
            with mock.patch.dict(os.environ, {"DAYDREAM_HOME": str(home)}, clear=False):
                reload_module("daydream.config")
                storage = reload_module("daydream.storage")

                first = storage.ChatSession(
                    session_id="older",
                    model="one",
                    title="older",
                    created_at=1.0,
                    updated_at=10.0,
                )
                second = storage.ChatSession(
                    session_id="newer",
                    model="two",
                    title="newer",
                    created_at=2.0,
                    updated_at=20.0,
                )
                storage.save_session(first)
                storage.save_session(second)

                sessions = storage.list_sessions()
                self.assertEqual([session.session_id for session in sessions], ["newer", "older"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from unittest import mock

from daydream import dreaming
from daydream.storage import ChatMessage, Memory


class DreamingTests(unittest.TestCase):
    def test_run_reming_returns_memory_objects(self) -> None:
        with mock.patch(
            "daydream.dreaming._call_model_for_memories",
            return_value=[
                {
                    "content": "User prefers short answers.",
                    "category": "preference",
                    "importance": 0.8,
                }
            ],
        ):
            memories = dreaming.run_reming(
                "model",
                "tokenizer",
                [ChatMessage(role="user", content="Be concise.", timestamp=1.0)],
            )

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].source_phase, "reming")
        self.assertEqual(memories[0].category, "preference")

    def test_run_dreaming_uses_current_session_existing_memories(self) -> None:
        phase_inputs: list[str] = []

        def fake_call(_model, _tokenizer, system_prompt, user_content, **_kwargs):
            phase_inputs.append(user_content)
            if system_prompt == dreaming.N3_SWS_SYSTEM:
                return [{"content": "User is building Daydream.", "category": "context"}]
            if system_prompt == dreaming.N2_SPINDLE_SYSTEM:
                return [{"content": "User is building Daydream.", "category": "context", "importance": 0.7}]
            return [{"content": "User is building Daydream.", "category": "insight", "importance": 0.9}]

        current_session_memories = [
            Memory(
                content="User prefers local-first tooling.",
                category="preference",
                importance=0.95,
                source_phase="rem",
                session_id="current",
            )
        ]

        with mock.patch("daydream.dreaming._call_model_for_memories", side_effect=fake_call), \
            mock.patch("daydream.dreaming.load_all_memories") as load_all_memories:
            memories = dreaming.run_dreaming(
                "model",
                "tokenizer",
                [ChatMessage(role="user", content="Remember I prefer local tooling.", timestamp=1.0)],
                "current",
                existing_memories=current_session_memories,
            )

        load_all_memories.assert_not_called()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].source_phase, "rem")
        self.assertIn("User prefers local-first tooling.", phase_inputs[-1])

    def test_run_dreaming_falls_back_to_n2_or_n3_source_phase(self) -> None:
        with mock.patch(
            "daydream.dreaming._call_model_for_memories",
            side_effect=[
                [{"content": "raw", "category": "fact"}],
                [{"content": "filtered", "category": "fact", "importance": 0.4}],
                [],
            ],
        ):
            memories = dreaming.run_dreaming(
                "model",
                "tokenizer",
                [ChatMessage(role="user", content="remember this", timestamp=1.0)],
                "session",
                existing_memories=[],
            )
        self.assertEqual(memories[0].source_phase, "n2")

        with mock.patch(
            "daydream.dreaming._call_model_for_memories",
            side_effect=[
                [{"content": "raw", "category": "fact"}],
                [],
                [],
            ],
        ):
            memories = dreaming.run_dreaming(
                "model",
                "tokenizer",
                [ChatMessage(role="user", content="remember this", timestamp=1.0)],
                "session",
                existing_memories=[],
            )
        self.assertEqual(memories[0].source_phase, "n3")


if __name__ == "__main__":
    unittest.main()

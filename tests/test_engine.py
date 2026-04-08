from __future__ import annotations

import types
import unittest
from unittest import mock

from daydream import engine


class EngineTests(unittest.TestCase):
    def test_generate_stream_passes_chat_template_kwargs(self) -> None:
        tokenizer = mock.Mock()
        tokenizer.apply_chat_template.return_value = "prompt"
        model = object()
        response = object()
        fake_mlx_lm = types.SimpleNamespace(stream_generate=lambda *args, **kwargs: iter([response]))
        fake_sample_utils = types.SimpleNamespace(make_sampler=lambda **kwargs: "sampler")

        with mock.patch.dict(
            "sys.modules",
            {
                "mlx_lm": fake_mlx_lm,
                "mlx_lm.sample_utils": fake_sample_utils,
            },
        ):
            outputs = list(
                engine.generate_stream(
                    model,
                    tokenizer,
                    [{"role": "user", "content": "hello"}],
                    chat_template_kwargs={"enable_thinking": False},
                )
            )

        self.assertEqual(outputs, [response])
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "hello"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )


if __name__ == "__main__":
    unittest.main()

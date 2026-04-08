"""Inference engine — thin wrapper over mlx-lm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional

from daydream.models import ensure_runtime_model, get_model_path, is_fixture_model

# Module-level cache for loaded model
_loaded_model = None
_loaded_tokenizer = None
_loaded_name: Optional[str] = None


@dataclass
class FixtureModel:
    repo_id: str


@dataclass
class FixtureTokenizer:
    repo_id: str

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        parts = [f"{message['role']}: {message.get('content', '')}" for message in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        prompt = "\n".join(parts)
        if tokenize:
            return prompt.split()
        return prompt


@dataclass
class FixtureResponse:
    text: str
    token: int
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: Optional[str]


def _fixture_reply(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", "")).lower()
        if "hello" in content:
            return "Hello from Daydream."
        return "Daydream is running in offline fixture mode."
    return "Hello from Daydream."


def load_model(name: str, verbose: bool = False, *, ensure_available: bool = True):
    """Load a model by short name or HF repo ID. Caches the result."""
    global _loaded_model, _loaded_tokenizer, _loaded_name

    repo_id = (
        ensure_runtime_model(name, auto_pull=True, register_alias=True)
        if ensure_available
        else name
    )
    if _loaded_name == repo_id and _loaded_model is not None:
        return _loaded_model, _loaded_tokenizer

    if is_fixture_model(repo_id):
        _loaded_model = FixtureModel(repo_id)
        _loaded_tokenizer = FixtureTokenizer(repo_id)
        _loaded_name = repo_id
        return _loaded_model, _loaded_tokenizer

    if verbose:
        from rich.console import Console

        Console(stderr=True).print(f"[dim]Loading {repo_id}...[/dim]")

    from mlx_lm import load as mlx_load

    model_ref = str(get_model_path(repo_id) or repo_id)
    _loaded_model, _loaded_tokenizer = mlx_load(model_ref)
    _loaded_name = repo_id
    return _loaded_model, _loaded_tokenizer


def generate_stream(
    model,
    tokenizer,
    messages: list[dict],
    *,
    max_tokens: int = 4096,
    temp: float = 0.6,
    top_p: float = 0.9,
    chat_template_kwargs: dict | None = None,
) -> Generator:
    """Stream-generate a response from a list of chat messages."""
    if isinstance(model, FixtureModel):
        text = _fixture_reply(messages)
        yield FixtureResponse(
            text=text[:max_tokens],
            token=0,
            prompt_tokens=sum(len(str(m.get("content", "")).split()) for m in messages),
            prompt_tps=1.0,
            generation_tokens=len(text.split()),
            generation_tps=1.0,
            peak_memory=0.0,
            finish_reason="stop",
        )
        return

    # Apply chat template to get the prompt string.
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )
    else:
        prompt = "\n".join(f"{message['role']}: {message.get('content', '')}" for message in messages)

    from mlx_lm import stream_generate as mlx_stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temp, top_p=top_p)

    for response in mlx_stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        yield response

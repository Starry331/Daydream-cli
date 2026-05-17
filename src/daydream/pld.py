"""Prompt-lookup decoding (PLD) — model-free speculative decoding.

Idea: when the model is generating tokens that mirror something already in
the prompt (e.g. code refactors, document edits, citation completion), we
can propose the *next-after-match* tokens from the prompt as a "draft" and
verify them in a single forward pass through the main model.

No second model is loaded. No extra memory. Worst case (no match found)
the loop degrades to standard greedy/sampled decoding with no overhead.

Reference: "LLM speculative decoding via prompt lookup" — Yao Fu et al.,
Apoorv Saxena's PLD implementation in transformers.

Limitations on Qwen3.5 / Qwen3.6 hybrid models:
    GatedDeltaNet's ArraysCache cannot be trimmed cleanly. Rejected draft
    tokens leave state behind that corrupts subsequent predictions. We
    detect this case at runtime (cache.can_trim_prompt_cache) and refuse
    to run PLD on it — the caller is expected to fall back to normal
    decoding instead of silently producing garbage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generator, Optional

import mlx.core as mx


@dataclass
class PLDConfig:
    """Configuration knobs for prompt-lookup decoding.

    `ngram_size`: how many trailing tokens must match exactly in the
        prompt for us to propose a continuation. Smaller = more matches
        but more rejections. 3 is the sweet spot per the original paper.
    `max_speculative`: cap on candidate tokens proposed per step.
    `min_ngram_size`: if no match at ngram_size, try shorter ngrams down
        to this floor (helps with shorter contexts).
    """

    ngram_size: int = 3
    min_ngram_size: int = 2
    max_speculative: int = 5


def _find_ngram_match(
    haystack: list[int],
    needle: list[int],
    last_emitted_index: int,
) -> Optional[int]:
    """Return the position in `haystack` where `needle` is found, or None.

    We scan from the start so the earliest (and longest-context) match
    wins. `last_emitted_index` is the position of the last token we've
    actually emitted — we won't propose tokens that come from BEFORE
    that point (they're already in the past) but we WILL look back into
    the prompt itself.
    """
    n = len(needle)
    if n == 0 or len(haystack) < n + 1:
        return None
    # Search up to (but not including) the trailing instance of needle —
    # that trailing instance IS our recent context.
    end = len(haystack) - n
    for i in range(0, end):
        if haystack[i : i + n] == needle:
            return i + n  # position right AFTER the match
    return None


def supports_pld(model) -> tuple[bool, str | None]:
    """Check whether PLD is safe on this model.

    PLD requires the prompt cache to be trimmable so we can revert
    rejected candidate tokens. Hybrid models (Qwen3.5 / Qwen3.6) have
    GatedDeltaNet recurrent state that isn't cleanly trimmable.
    """
    try:
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache
    except ImportError as exc:
        return False, f"mlx-lm not available: {exc}"
    if not hasattr(model, "__call__"):
        return False, "Model is not callable."
    try:
        probe = make_prompt_cache(model)
    except Exception as exc:
        return False, f"Could not build prompt cache: {exc}"
    if not can_trim_prompt_cache(probe):
        return False, (
            "Model's prompt cache is not trimmable (hybrid GatedDeltaNet "
            "recurrent state). PLD would corrupt output."
        )
    return True, None


def pld_stream(
    model,
    tokenizer,
    prompt: str,
    *,
    max_tokens: int,
    sampler,
    config: PLDConfig | None = None,
    prompt_cache=None,
) -> Generator:
    """Stream-generate via prompt-lookup speculative decoding.

    Yields `mlx_lm.generate.GenerationResponse`-shaped objects with the
    same fields the rest of Daydream expects (`text`, `token`,
    `prompt_tokens`, `generation_tokens`, `generation_tps`,
    `prompt_tps`, `peak_memory`, `finish_reason`, `from_draft`).

    This is a custom generation loop, not a wrapper around mlx-lm's
    `stream_generate`, because mlx-lm has no PLD primitive. We rebuild
    the same streaming contract here so the rest of the daydream
    pipeline (Rich UI, /effort, /context, server SSE) doesn't have to
    know the difference.
    """
    cfg = config or PLDConfig()

    from mlx_lm.generate import GenerationResponse
    from mlx_lm.models.cache import (
        can_trim_prompt_cache,
        make_prompt_cache,
        trim_prompt_cache,
    )

    prompt_ids: list[int] = tokenizer.encode(prompt)
    if not prompt_ids:
        return

    if prompt_cache is None:
        cache = make_prompt_cache(model)
    else:
        cache = prompt_cache

    if not can_trim_prompt_cache(cache):
        raise ValueError(
            "PLD requires a trimmable prompt cache. Switch off /draft "
            "lookup for this model family — Qwen3.5 / Qwen3.6 hybrid "
            "models are not yet supported."
        )

    # ── Phase 1: prefill ────────────────────────────────────────────
    prefill_tic = time.perf_counter()
    prompt_arr = mx.array([prompt_ids])
    logits = model(prompt_arr, cache=cache)
    logits = logits[:, -1, :]
    # Decode the first emitted token from the prefill logits.
    first_token = int(sampler(logits).item())
    prefill_time = time.perf_counter() - prefill_tic
    prompt_tps = (len(prompt_ids) / prefill_time) if prefill_time > 0 else 0.0

    eos_ids: set[int] = set()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, int):
        eos_ids.add(eos_id)
    eos_ids_list = getattr(tokenizer, "eos_token_ids", None)
    if isinstance(eos_ids_list, (list, tuple)):
        eos_ids.update(int(t) for t in eos_ids_list if isinstance(t, int))

    generated: list[int] = [first_token]
    detok = _IncrementalDetokenizer(tokenizer)
    first_text = detok.add(first_token)

    gen_tic = time.perf_counter()
    yield _make_response(
        text=first_text,
        token=first_token,
        prompt_tokens=len(prompt_ids),
        prompt_tps=prompt_tps,
        generation_tokens=1,
        generation_tps=1.0 / max(time.perf_counter() - gen_tic, 1e-9),
        peak_memory=0.0,
        finish_reason="stop" if first_token in eos_ids else None,
        from_draft=False,
        GenerationResponse=GenerationResponse,
    )
    if first_token in eos_ids:
        return

    # ── Phase 2: speculative loop ───────────────────────────────────
    ntok = 1
    while ntok < max_tokens:
        # Pick the best ngram match (largest n first).
        candidates: list[int] = []
        haystack = prompt_ids + generated
        for n in range(cfg.ngram_size, cfg.min_ngram_size - 1, -1):
            if len(generated) < n:
                continue
            needle = generated[-n:]
            pos = _find_ngram_match(haystack, needle, last_emitted_index=len(prompt_ids) - 1)
            if pos is None:
                continue
            # Take up to max_speculative tokens after the match.
            candidates = haystack[pos : pos + cfg.max_speculative]
            if candidates:
                break

        if not candidates:
            # No match — single-step decode.
            inp = mx.array([[generated[-1]]])
            logits = model(inp, cache=cache)
            next_tok = int(sampler(logits[:, -1, :]).item())
            generated.append(next_tok)
            text = detok.add(next_tok)
            ntok += 1
            yield _make_response(
                text=text, token=next_tok,
                prompt_tokens=len(prompt_ids),
                prompt_tps=prompt_tps,
                generation_tokens=ntok,
                generation_tps=ntok / max(time.perf_counter() - gen_tic, 1e-9),
                peak_memory=0.0,
                finish_reason="stop" if next_tok in eos_ids else None,
                from_draft=False,
                GenerationResponse=GenerationResponse,
            )
            if next_tok in eos_ids:
                return
            continue

        # ── Verify candidates in one forward pass ──────────────────
        # Feed [last_emitted, c0, c1, ..., cK-1]. The logits at each
        # position predict the *following* token.
        verify_seq = [generated[-1]] + candidates
        verify_arr = mx.array([verify_seq])
        logits = model(verify_arr, cache=cache)
        # logits shape: (1, K+1, vocab)
        # We want predictions at positions 0..K-1 — each predicting
        # what comes after [last_emitted + c0..ci-1].
        # Compare argmax / sampled to candidates[i].
        predicted = sampler(logits[0])  # shape: (K+1,)
        predicted_list = [int(x) for x in predicted.tolist()]

        accepted: list[int] = []
        bumped: int | None = None
        for i, cand in enumerate(candidates):
            if predicted_list[i] == cand:
                accepted.append(cand)
            else:
                bumped = predicted_list[i]
                break
        else:
            # All candidates accepted. The model's prediction at the
            # *last* verify position (K) gives the next free token.
            bumped = predicted_list[len(candidates)]

        # Cache currently has K+1 candidate positions added. We want it
        # to reflect: last_emitted + accepted prefix only (the bumped
        # token will be the START of the next iteration, fed through
        # the model on its own).
        # That's 1 + len(accepted) positions to KEEP; the rest are
        # rejected. Trim by (K+1) - (1 + len(accepted)) = K - len(accepted).
        trim_n = len(candidates) - len(accepted)
        if trim_n > 0:
            trim_prompt_cache(cache, trim_n)

        # Emit accepted tokens (from draft) then the bumped token (not).
        for tok in accepted:
            generated.append(tok)
            text = detok.add(tok)
            ntok += 1
            yield _make_response(
                text=text, token=tok,
                prompt_tokens=len(prompt_ids),
                prompt_tps=prompt_tps,
                generation_tokens=ntok,
                generation_tps=ntok / max(time.perf_counter() - gen_tic, 1e-9),
                peak_memory=0.0,
                finish_reason="stop" if tok in eos_ids else None,
                from_draft=True,
                GenerationResponse=GenerationResponse,
            )
            if tok in eos_ids:
                return
            if ntok >= max_tokens:
                return

        # The bumped token is the model's own prediction after the
        # accepted prefix. Emit it as a non-draft token.
        if bumped is not None and ntok < max_tokens:
            generated.append(bumped)
            text = detok.add(bumped)
            ntok += 1
            yield _make_response(
                text=text, token=bumped,
                prompt_tokens=len(prompt_ids),
                prompt_tps=prompt_tps,
                generation_tokens=ntok,
                generation_tps=ntok / max(time.perf_counter() - gen_tic, 1e-9),
                peak_memory=0.0,
                finish_reason="stop" if bumped in eos_ids else None,
                from_draft=False,
                GenerationResponse=GenerationResponse,
            )
            if bumped in eos_ids:
                return


def _make_response(*, text, token, prompt_tokens, prompt_tps,
                   generation_tokens, generation_tps, peak_memory,
                   finish_reason, from_draft, GenerationResponse):
    return GenerationResponse(
        text=text,
        token=token,
        logprobs=None,
        from_draft=from_draft,
        prompt_tokens=prompt_tokens,
        prompt_tps=prompt_tps,
        generation_tokens=generation_tokens,
        generation_tps=generation_tps,
        peak_memory=peak_memory,
        finish_reason=finish_reason,
    )


class _IncrementalDetokenizer:
    """Token-id → text stream that handles BPE / SentencePiece boundaries.

    mlx-lm has a `naive_streaming_detokenizer` helper but it's not part
    of the public API across versions. We implement a minimal-good-
    enough version inline so we don't bind to a private symbol.
    """

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.tokens: list[int] = []
        self.last_text: str = ""

    def add(self, token: int) -> str:
        self.tokens.append(token)
        try:
            text = self.tokenizer.decode(self.tokens, skip_special_tokens=False)
        except TypeError:
            text = self.tokenizer.decode(self.tokens)
        delta = text[len(self.last_text):]
        self.last_text = text
        return delta

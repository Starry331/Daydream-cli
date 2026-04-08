"""Memory consolidation engine — neuroscience-inspired dreaming phases."""

from __future__ import annotations

import json
import time
from typing import Callable

from daydream import engine
from daydream.storage import ChatMessage, Memory, load_all_memories

# ── System prompts ─────────────────────────────────────────────────────

REMING_SYSTEM = (
    "You are a memory extraction assistant. Analyze the conversation and extract important memories.\n"
    "For each memory, output a JSON object on its own line:\n"
    '{"content": "...", "category": "fact|preference|context|insight", "importance": 0.0-1.0}\n'
    "\n"
    "Extract: facts learned, user preferences, important context, key decisions, useful insights.\n"
    "Discard: greetings, filler, obvious/trivial exchanges, repetitive content.\n"
    "Output ONLY the JSON lines, nothing else."
)

N3_SWS_SYSTEM = (
    "You are performing hippocampal replay (N3/SWS deep sleep phase).\n"
    "Replay the conversation and extract ALL declarative memories — facts, events, statements, decisions.\n"
    "Be thorough and exhaustive. Include everything that could potentially be important.\n"
    "For each memory, output a JSON object on its own line:\n"
    '{"content": "...", "category": "fact|preference|context|insight"}\n'
    "Output ONLY the JSON lines."
)

N2_SPINDLE_SYSTEM = (
    "You are performing sleep spindle processing (N2/Core sleep phase).\n"
    "You will receive a list of candidate memories. Your job is to:\n"
    "1. DISCARD trivial, redundant, or obvious memories\n"
    "2. MERGE overlapping memories into consolidated ones\n"
    "3. RATE each surviving memory's importance (0.0-1.0)\n"
    "\n"
    "For each memory to keep, output a JSON object on its own line:\n"
    '{"content": "...", "category": "fact|preference|context|insight", "importance": 0.0-1.0}\n'
    "Output ONLY the JSON lines."
)

REM_INTEGRATION_SYSTEM = (
    "You are performing REM sleep memory integration.\n"
    "You will receive:\n"
    "1. NEW memories from the current session (after N2 filtering)\n"
    "2. EXISTING memories from previous sessions\n"
    "\n"
    "Your job is to:\n"
    "1. CONNECT new memories to existing ones where relevant\n"
    "2. FIND PATTERNS across sessions\n"
    "3. CREATE abstract insights from concrete memories\n"
    "4. REWEIGHT importance based on cross-session relevance\n"
    "\n"
    "For each final memory, output a JSON object on its own line:\n"
    '{"content": "...", "category": "fact|preference|context|insight|pattern", "importance": 0.0-1.0}\n'
    "Output ONLY the JSON lines."
)


# ── Core functions ─────────────────────────────────────────────────────

def _format_conversation(messages: list[ChatMessage]) -> str:
    """Format messages into readable text for the model."""
    parts: list[str] = []
    for msg in messages:
        role = msg.role.capitalize()
        parts.append(f"{role}: {msg.content}")
    return "\n\n".join(parts)


def _format_memories_for_input(memories: list[dict]) -> str:
    """Format raw memory dicts as text for the next phase."""
    lines: list[str] = []
    for mem in memories:
        lines.append(f"- [{mem.get('category', 'fact')}] {mem.get('content', '')}")
    return "\n".join(lines)


def _format_new_and_existing(
    new_memories: list[dict],
    existing_memories: list[Memory],
) -> str:
    """Format new and existing memories for REM integration."""
    parts: list[str] = []
    parts.append("=== NEW MEMORIES (current session) ===")
    for mem in new_memories:
        importance = mem.get("importance", 0.5)
        parts.append(f"- [{mem.get('category', 'fact')}] (importance: {importance:.1f}) {mem.get('content', '')}")

    if existing_memories:
        parts.append("")
        parts.append("=== EXISTING MEMORIES (previous sessions) ===")
        for mem in existing_memories:
            parts.append(f"- [{mem.category}] (importance: {mem.importance:.1f}) {mem.content}")

    return "\n".join(parts)


def _parse_memory_json_lines(text: str) -> list[dict]:
    """Parse model output as JSON lines, skip malformed lines."""
    results: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Try to extract JSON from the line
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "content" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            # Try to find JSON within the line
            start = line.find("{")
            end = line.rfind("}")
            if start != -1 and end > start:
                try:
                    obj = json.loads(line[start:end + 1])
                    if isinstance(obj, dict) and "content" in obj:
                        results.append(obj)
                except json.JSONDecodeError:
                    continue
    return results


def _call_model_for_memories(
    model,
    tokenizer,
    system_prompt: str,
    user_content: str,
    *,
    temp: float,
    max_tokens: int,
    on_token: Callable[[str], None] | None = None,
) -> list[dict]:
    """Run a single generation pass with the given system prompt.

    Parse output as JSON-lines. Return list of memory dicts.
    on_token callback used for streaming animation updates.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    full_text = ""
    for response in engine.generate_stream(
        model, tokenizer, messages,
        max_tokens=max_tokens, temp=temp,
    ):
        chunk = response.text or ""
        full_text += chunk
        if on_token is not None and chunk:
            on_token(chunk)
        if response.finish_reason:
            break

    return _parse_memory_json_lines(full_text)


def run_reming(
    model,
    tokenizer,
    messages: list[ChatMessage],
    *,
    temp: float = 0.3,
    max_tokens: int = 2048,
    on_phase: Callable[[str, str], None] | None = None,
    on_token: Callable[[str], None] | None = None,
) -> list[Memory]:
    """Single-pass quick memory extraction.

    1. Format conversation as text
    2. Call model with REMING_SYSTEM
    3. Parse JSON lines -> Memory objects with source_phase="reming"
    """
    if on_phase is not None:
        on_phase("reming", "REMing — extracting memories...")

    conversation_text = _format_conversation(messages)
    raw = _call_model_for_memories(
        model, tokenizer, REMING_SYSTEM, conversation_text,
        temp=temp, max_tokens=max_tokens, on_token=on_token,
    )

    now = time.time()
    memories: list[Memory] = []
    for item in raw:
        memories.append(Memory(
            content=item.get("content", ""),
            category=item.get("category", "fact"),
            importance=float(item.get("importance", 0.5)),
            source_phase="reming",
            created_at=now,
            session_id="",
        ))
    return memories


def run_dreaming(
    model,
    tokenizer,
    messages: list[ChatMessage],
    session_id: str,
    existing_memories: list[Memory] | None = None,
    *,
    temp: float = 0.3,
    max_tokens: int = 2048,
    on_phase: Callable[[str, str], None] | None = None,
    on_token: Callable[[str], None] | None = None,
) -> list[Memory]:
    """Full 3-phase sleep cycle.

    Phase 1 — N3/SWS: Hippocampal replay — extract raw declarative memories
    Phase 2 — N2/Core: Spindle processing — select, gate, filter
    Phase 3 — REM: Integration & abstraction — connect, pattern-find, reweight
    """
    conversation_text = _format_conversation(messages)

    # Phase 1 — N3/SWS
    if on_phase is not None:
        on_phase("n3", "Deep sleep — hippocampal replay...")
    raw_memories = _call_model_for_memories(
        model, tokenizer, N3_SWS_SYSTEM, conversation_text,
        temp=temp, max_tokens=max_tokens, on_token=on_token,
    )

    if not raw_memories:
        return []

    # Phase 2 — N2/Core
    if on_phase is not None:
        on_phase("n2", "Core sleep — spindle processing...")
    formatted_raw = _format_memories_for_input(raw_memories)
    filtered = _call_model_for_memories(
        model, tokenizer, N2_SPINDLE_SYSTEM, formatted_raw,
        temp=temp, max_tokens=max_tokens, on_token=on_token,
    )

    source_phase = "n2"
    if not filtered:
        filtered = raw_memories
        source_phase = "n3"

    # Phase 3 — REM
    if on_phase is not None:
        on_phase("rem", "REM sleep — integration...")
    existing = existing_memories if existing_memories is not None else load_all_memories()
    all_context = _format_new_and_existing(filtered, existing)
    final = _call_model_for_memories(
        model, tokenizer, REM_INTEGRATION_SYSTEM, all_context,
        temp=temp, max_tokens=max_tokens, on_token=on_token,
    )

    if not final:
        final = filtered
    else:
        source_phase = "rem"

    now = time.time()
    memories: list[Memory] = []
    for item in final:
        memories.append(Memory(
            content=item.get("content", ""),
            category=item.get("category", "fact"),
            importance=float(item.get("importance", 0.5)),
            source_phase=source_phase,
            created_at=now,
            session_id=session_id,
        ))
    return memories

"""Memory consolidation engine — neuroscience-inspired dreaming phases."""

from __future__ import annotations

import json
import time
from typing import Callable

from daydream import engine
from daydream.storage import ChatMessage, Memory, load_all_memories

# ── System prompts ─────────────────────────────────────────────────────

REMING_SYSTEM = (
    "You are performing REM-only memory processing — representational reformatting of a conversation.\n"
    "Unlike systematic consolidation, you operate without prior stabilization. "
    "Your processing should be creative, associative, and transformative.\n\n"
    "Perform these operations on the conversation:\n"
    "1. ASSOCIATIVE RECOMBINATION: Link scattered ideas and facts into new connections "
    "that were not explicitly stated. Find hidden relationships.\n"
    "2. SCHEMA ABSTRACTION: Distill specific exchanges into general principles, preferences, "
    "or conceptual frameworks the user operates within.\n"
    "3. AFFECTIVE REWEIGHTING: Identify what matters most to the user emotionally or motivationally. "
    "Elevate important concerns, demote trivial ones.\n"
    "4. GENERALIZATION: Sacrifice verbatim detail for higher-order patterns and transferable insights.\n\n"
    "For each memory, output a JSON object on its own line:\n"
    '{"content": "...", "category": "fact|preference|context|insight|pattern", "importance": 0.0-1.0}\n\n'
    "Categories: fact=concrete knowledge, preference=user likes/dislikes/style, "
    "context=situational background, insight=derived understanding, pattern=cross-topic regularity.\n"
    "Importance: 0.0=trivial, 0.5=moderately useful, 1.0=critical to future interactions.\n"
    "Output ONLY the JSON lines, nothing else."
)

N2_SPINDLE_SYSTEM = (
    "You are performing N2 sleep spindle processing — the gating phase of memory consolidation.\n"
    "Your role is SELECTION and FILTERING, not extraction. You decide which experiences "
    "from this conversation deserve to be consolidated into long-term memory.\n\n"
    "Apply these gating criteria:\n"
    "1. REACTIVATION WINDOWING: Identify moments in the conversation that carry learning signal "
    "— new information, corrections, decisions, or preference reveals.\n"
    "2. SENSORY GATING: Filter out noise — greetings, filler, small talk, repetitive exchanges, "
    "and anything that does not contain novel or useful information.\n"
    "3. INTERFERENCE SHIELDING: If multiple similar memories exist, select the most precise or "
    "recent version. Remove contradictions by keeping the latest truth.\n"
    "4. PLASTICITY TIMING: Prioritize memories that would be most useful for future conversations "
    "— things the user is likely to reference again or that change how you should respond.\n\n"
    "For each memory that passes your gate, output a JSON object on its own line:\n"
    '{"content": "...", "category": "fact|preference|context|insight"}\n'
    "Output ONLY the JSON lines."
)

N3_SWS_SYSTEM = (
    "You are performing N3/SWS deep sleep processing — stabilization and systematic consolidation.\n"
    "You receive memories that passed N2 gating. Your job is to STABILIZE, ORGANIZE, and PRUNE.\n\n"
    "Apply these consolidation operations:\n"
    "1. TRACE STABILIZATION: Refine each memory into a clear, self-contained statement. "
    "Remove ambiguity. Ensure each memory is understandable without conversation context.\n"
    "2. REDISTRIBUTION: Reorganize memories into proper categories. Facts should be factual, "
    "preferences should capture user intent, contexts should be situational.\n"
    "3. NOISE PRUNING: Merge near-duplicate memories. Tighten wording. Remove any remaining "
    "filler that slipped through N2 gating.\n"
    "4. CAPACITY RESET: Rate each memory importance (0.0-1.0) based on predicted future utility. "
    "Be strict — only truly important memories should score above 0.7.\n\n"
    "For each stabilized memory, output a JSON object on its own line:\n"
    '{"content": "...", "category": "fact|preference|context|insight", "importance": 0.0-1.0}\n'
    "Output ONLY the JSON lines."
)

REM_INTEGRATION_SYSTEM = (
    "You are performing REM sleep integration — the final phase after NREM stabilization.\n"
    "You will receive:\n"
    "1. NEW memories: stabilized by N3 from the current session\n"
    "2. EXISTING memories: from previous sessions\n\n"
    "Because these memories were already filtered (N2) and stabilized (N3), your integration "
    "should be CONSTRAINED — build on the stable foundation rather than free-associating.\n\n"
    "Perform these operations:\n"
    "1. INTEGRATION: Connect new memories to existing ones. If a new fact extends or updates "
    "an existing memory, merge them into one richer memory.\n"
    "2. ABSTRACTION: Where multiple specific memories point to the same pattern, create a "
    "higher-level insight that captures the underlying principle.\n"
    "3. EMOTIONAL RECALIBRATION: Re-evaluate importance scores in light of the full memory set. "
    "A memory that seemed minor alone may become important when it connects to a pattern.\n"
    "4. REPRESENTATIONAL TRANSFORMATION: Rewrite memories to be maximally useful for an AI "
    "assistant — focus on actionable knowledge, user preferences, and contextual cues.\n\n"
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

    Phase 1 — N3/SWS: Deep sleep — extract raw declarative memories
    Phase 2 — N2/Core: Spindle processing — select and gate candidate memories
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
        on_phase("n2", "Core sleep — spindle gating...")
    formatted_raw = _format_memories_for_input(raw_memories)
    stabilized = _call_model_for_memories(
        model, tokenizer, N2_SPINDLE_SYSTEM, formatted_raw,
        temp=temp, max_tokens=max_tokens, on_token=on_token,
    )

    phase_memories = stabilized
    source_phase = "n2"
    if not phase_memories:
        phase_memories = raw_memories
        source_phase = "n3"

    # Phase 3 — REM
    if on_phase is not None:
        on_phase("rem", "REM sleep — integration...")
    existing = existing_memories if existing_memories is not None else load_all_memories()
    all_context = _format_new_and_existing(phase_memories, existing)
    final = _call_model_for_memories(
        model, tokenizer, REM_INTEGRATION_SYSTEM, all_context,
        temp=temp, max_tokens=max_tokens, on_token=on_token,
    )

    if not final:
        final = phase_memories
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

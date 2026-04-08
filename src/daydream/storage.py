"""Persistence layer for chat sessions and memories."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from daydream.config import CHATS_DIR, MEMORIES_DIR, ensure_home


@dataclass
class ChatMessage:
    role: str           # "user" | "assistant" | "system"
    content: str
    timestamp: float    # time.time()
    reasoning: str = "" # captured reasoning text (if any)


@dataclass
class Memory:
    content: str        # the memory text
    category: str       # "fact" | "preference" | "context" | "insight" | "pattern"
    importance: float   # 0.0-1.0, set during N2 phase (or REMing)
    source_phase: str   # "reming" | "n3" | "n2" | "rem"
    created_at: float = 0.0
    session_id: str = ""  # which session produced this


@dataclass
class ChatSession:
    session_id: str     # uuid4 hex
    model: str          # model short name
    title: str          # auto-generated or user-set, from first message
    created_at: float
    updated_at: float
    messages: list[ChatMessage] = field(default_factory=list)
    memories: list[Memory] = field(default_factory=list)


# ── Session I/O ────────────────────────────────────────────────────────

def save_session(session: ChatSession) -> None:
    """Write session JSON to CHATS_DIR / f"{session.session_id}.json".

    Auto-generates title from first user message if title is empty.
    """
    ensure_home()

    if not session.title:
        for msg in session.messages:
            if msg.role == "user" and msg.content.strip():
                session.title = msg.content.strip()[:50]
                break

    path = CHATS_DIR / f"{session.session_id}.json"
    data = asdict(session)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_session(session_id: str) -> ChatSession | None:
    """Load a single session by ID, or return None if not found."""
    path = CHATS_DIR / f"{session_id}.json"
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    return _session_from_dict(data)


def list_sessions() -> list[ChatSession]:
    """Scan CHATS_DIR, load metadata (messages truncated for listing).

    Returns sessions sorted by updated_at descending.
    """
    ensure_home()
    sessions: list[ChatSession] = []

    if not CHATS_DIR.exists():
        return sessions

    for path in CHATS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        session = _session_from_dict(data)
        if session is not None:
            sessions.append(session)

    sessions.sort(key=lambda s: s.updated_at, reverse=True)
    return sessions


def delete_session(session_id: str) -> bool:
    """Delete a session file. Returns True if it existed."""
    path = CHATS_DIR / f"{session_id}.json"
    memory_path = MEMORIES_DIR / f"{session_id}.json"
    if path.exists():
        path.unlink()
        if memory_path.exists():
            memory_path.unlink()
        return True
    return False


# ── Memory I/O ─────────────────────────────────────────────────────────

def save_memories(session_id: str, memories: list[Memory]) -> None:
    """Write memories to MEMORIES_DIR / f"{session_id}.json"."""
    ensure_home()
    path = MEMORIES_DIR / f"{session_id}.json"
    data = [asdict(m) for m in memories]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_memories(session_id: str) -> list[Memory]:
    """Load memories for a single session."""
    path = MEMORIES_DIR / f"{session_id}.json"
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []

    memories: list[Memory] = []
    for item in data:
        mem = _memory_from_dict(item)
        if mem is not None:
            memories.append(mem)
    return memories


def load_all_memories() -> list[Memory]:
    """Aggregate all memory files — used by REM phase for cross-session integration."""
    ensure_home()
    all_memories: list[Memory] = []

    if not MEMORIES_DIR.exists():
        return all_memories

    for path in MEMORIES_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            mem = _memory_from_dict(item)
            if mem is not None:
                all_memories.append(mem)

    return all_memories


# ── Internal helpers ───────────────────────────────────────────────────

def _session_from_dict(data: dict) -> ChatSession | None:
    """Reconstruct a ChatSession from a JSON-loaded dict."""
    try:
        messages = [
            ChatMessage(
                role=m["role"],
                content=m["content"],
                timestamp=m.get("timestamp", 0.0),
                reasoning=m.get("reasoning", ""),
            )
            for m in data.get("messages", [])
        ]
        memories = []
        for m in data.get("memories", []):
            mem = _memory_from_dict(m)
            if mem is not None:
                memories.append(mem)
        return ChatSession(
            session_id=data["session_id"],
            model=data.get("model", ""),
            title=data.get("title", ""),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
            messages=messages,
            memories=memories,
        )
    except (KeyError, TypeError):
        return None


def _memory_from_dict(data: dict) -> Memory | None:
    """Reconstruct a Memory from a JSON-loaded dict."""
    try:
        return Memory(
            content=data["content"],
            category=data.get("category", "fact"),
            importance=float(data.get("importance", 0.5)),
            source_phase=data.get("source_phase", "reming"),
            created_at=data.get("created_at", 0.0),
            session_id=data.get("session_id", ""),
        )
    except (KeyError, TypeError, ValueError):
        return None

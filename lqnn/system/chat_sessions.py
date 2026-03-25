"""Persistent chat sessions storage for terminal UI."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

SESSIONS_FILE = Path("data/state/chat_sessions.json")


class ChatSessionStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or SESSIONS_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._sessions = {}
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._sessions = raw
            else:
                self._sessions = {}
        except Exception:
            self._sessions = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._sessions, ensure_ascii=True),
            encoding="utf-8",
        )

    def list_sessions(self) -> list[dict]:
        rows = []
        for s in self._sessions.values():
            rows.append({
                "id": s["id"],
                "title": s.get("title", "New chat"),
                "created_at": s.get("created_at", 0.0),
                "updated_at": s.get("updated_at", 0.0),
                "message_count": len(s.get("messages", [])),
            })
        rows.sort(key=lambda x: x.get("updated_at", 0.0), reverse=True)
        return rows

    def get_session(self, session_id: str) -> dict | None:
        return self._sessions.get(session_id)

    def create_session(self, title: str = "New chat") -> dict:
        sid = uuid.uuid4().hex
        now = time.time()
        sess = {
            "id": sid,
            "title": title.strip()[:80] or "New chat",
            "messages": [],
            "created_at": now,
            "updated_at": now,
        }
        self._sessions[sid] = sess
        self._save()
        return sess

    def upsert_session(
        self,
        session_id: str,
        title: str | None = None,
        messages: list[dict] | None = None,
    ) -> dict:
        now = time.time()
        existing = self._sessions.get(session_id)
        if not existing:
            existing = {
                "id": session_id,
                "title": (title or "New chat").strip()[:80] or "New chat",
                "messages": [],
                "created_at": now,
                "updated_at": now,
            }
            self._sessions[session_id] = existing
        if title is not None:
            existing["title"] = title.strip()[:80] or existing["title"]
        if messages is not None:
            existing["messages"] = messages[-200:]
        existing["updated_at"] = now
        self._save()
        return existing

    def delete_session(self, session_id: str) -> bool:
        if session_id not in self._sessions:
            return False
        del self._sessions[session_id]
        self._save()
        return True

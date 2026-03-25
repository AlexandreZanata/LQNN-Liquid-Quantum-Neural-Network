"""MongoDB integration for logging training cycles, chat turns, and agent activity."""

from __future__ import annotations

import logging
import os
import time

log = logging.getLogger(__name__)


class TrainingDB:
    """Persistent logging to MongoDB.

    Collections:
    - training_cycles: continuous trainer metrics
    - chat_turns: user chat interactions
    - agent_cycles: agent manager reports
    """

    def __init__(
        self,
        uri: str | None = None,
        db_name: str | None = None,
    ) -> None:
        self._uri = uri or os.environ.get("MONGO_URI", "mongodb://localhost:27017/lqnn")
        self._db_name = db_name or os.environ.get("MONGO_DB", "lqnn")
        self._client = None
        self._db = None
        self._available = False
        self._connect()

    def _connect(self) -> None:
        try:
            from pymongo import MongoClient
            self._client = MongoClient(self._uri, serverSelectionTimeoutMS=3000)
            self._client.admin.command("ping")
            self._db = self._client[self._db_name]
            self._ensure_indexes()
            self._available = True
            log.info("MongoDB connected: %s / %s", self._uri, self._db_name)
        except Exception as exc:
            log.warning("MongoDB unavailable (%s), logging disabled: %s",
                        self._uri, exc)
            self._available = False

    def _ensure_indexes(self) -> None:
        if not self._db:
            return
        self._db["training_cycles"].create_index("timestamp")
        self._db["training_cycles"].create_index("cycle")
        self._db["chat_turns"].create_index("created_at")
        self._db["agent_cycles"].create_index("timestamp")

    @property
    def available(self) -> bool:
        return self._available

    def log_training_cycle(self, data: dict) -> None:
        if not self._available:
            return
        try:
            data.setdefault("timestamp", time.time())
            self._db["training_cycles"].insert_one(data)
        except Exception:
            pass

    def log_chat_turn(self, user_text: str, response: str,
                      confidence: float, context_concepts: list[str] | None = None) -> None:
        if not self._available:
            return
        try:
            self._db["chat_turns"].insert_one({
                "user_text": user_text,
                "response": response,
                "confidence": confidence,
                "context_concepts": context_concepts or [],
                "created_at": time.time(),
            })
        except Exception:
            pass

    def log_agent_cycle(self, report: dict) -> None:
        if not self._available:
            return
        try:
            report.setdefault("timestamp", time.time())
            self._db["agent_cycles"].insert_one(report)
        except Exception:
            pass

    def get_recent_training(self, limit: int = 20) -> list[dict]:
        if not self._available:
            return []
        try:
            cursor = (
                self._db["training_cycles"]
                .find({}, {"_id": 0})
                .sort("timestamp", -1)
                .limit(limit)
            )
            return list(cursor)
        except Exception:
            return []

    def get_recent_chats(self, limit: int = 20) -> list[dict]:
        if not self._available:
            return []
        try:
            cursor = (
                self._db["chat_turns"]
                .find({}, {"_id": 0})
                .sort("created_at", -1)
                .limit(limit)
            )
            return list(cursor)
        except Exception:
            return []

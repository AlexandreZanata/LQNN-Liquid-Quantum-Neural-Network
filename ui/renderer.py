"""Serialize brain state for the UI."""

from __future__ import annotations

from typing import Any


def build_brain_payload(
    *,
    memory_stats: dict[str, Any] | None = None,
    training_status: dict[str, Any] | None = None,
    agent_status: dict[str, Any] | None = None,
    chat_history: list[dict[str, str]] | None = None,
    recent_concepts: list[dict] | None = None,
    recent_associations: list[dict] | None = None,
    training_log: list[dict] | None = None,
    agent_activity: list[dict] | None = None,
    system_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON payload pushed to UI clients via WebSocket."""
    return {
        "type": "state",
        "memory": memory_stats or {},
        "training": training_status or {},
        "agents": agent_status or {},
        "chat_history": chat_history or [],
        "recent_concepts": recent_concepts or [],
        "recent_associations": recent_associations or [],
        "training_log": training_log or [],
        "agent_activity": agent_activity or [],
        "system": system_metrics or {},
    }

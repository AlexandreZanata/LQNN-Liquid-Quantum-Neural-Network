"""UI controller -- bridges the web interface to the brain system."""

from __future__ import annotations

import logging
from typing import Any

from lqnn.core.associative_memory import AssociativeMemory
from lqnn.system.chat_engine import ChatEngine
from lqnn.training.continuous_trainer import ContinuousTrainer
from lqnn.agents.manager import AgentManager
from ui.renderer import build_brain_payload

log = logging.getLogger(__name__)


class UIController:
    """Maps UI actions to brain operations."""

    def __init__(
        self,
        memory: AssociativeMemory,
        chat_engine: ChatEngine,
        trainer: ContinuousTrainer,
        agent_manager: AgentManager,
    ) -> None:
        self.memory = memory
        self.chat_engine = chat_engine
        self.trainer = trainer
        self.agent_manager = agent_manager
        self.chat_history: list[dict[str, str]] = []

    def chat_turn(self, text: str) -> dict[str, Any]:
        result = self.chat_engine.chat(text, history=self.chat_history)
        self.chat_history.append({"role": "user", "text": text})
        self.chat_history.append({"role": "assistant", "text": result["response"]})
        self.chat_history = self.chat_history[-100:]
        return result

    async def search(self, query: str) -> dict[str, Any]:
        report = await self.agent_manager.request_search(query)
        return {
            "cycle": report.cycle,
            "concepts_learned": report.concepts_learned,
            "images_processed": report.images_processed,
            "duration_s": round(report.duration_s, 2),
        }

    async def run_agent_cycle(self) -> dict[str, Any]:
        report = await self.agent_manager.run_cycle()
        return {
            "cycle": report.cycle,
            "concepts_learned": report.concepts_learned,
            "images_processed": report.images_processed,
            "duration_s": round(report.duration_s, 2),
        }

    def consolidate(self) -> dict[str, Any]:
        return self.memory.consolidate()

    def self_play(self) -> dict[str, Any]:
        return self.memory.self_play_cycle()

    def learn_concept(self, concept: str) -> dict[str, Any]:
        state = self.memory.learn_concept(concept, source="ui_manual")
        return {
            "concept": state.concept,
            "associations": len(state.associations),
            "volatility": state.volatility,
        }

    async def train_cycle(self) -> dict[str, Any]:
        await self.trainer.run_manual_cycle()
        return self.trainer.latest_metrics() or {}

    def snapshot(self) -> dict[str, Any]:
        stable = self.memory.store.get_stable_concepts(threshold=0.3)
        recent_concepts = []
        for s in stable[:20]:
            meta = s.get("metadata", {})
            recent_concepts.append({
                "concept": s.get("document", ""),
                "volatility": meta.get("volatility", 1.0),
                "confidence": meta.get("confidence", 0.5),
                "access_count": meta.get("access_count", 0),
            })

        return build_brain_payload(
            memory_stats=self.memory.stats(),
            training_status=self.trainer.status(),
            agent_status=self.agent_manager.stats(),
            chat_history=self.chat_history[-20:],
            recent_concepts=recent_concepts,
        )

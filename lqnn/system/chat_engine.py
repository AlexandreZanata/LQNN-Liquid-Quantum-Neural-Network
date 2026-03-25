"""Chat engine -- answers user questions grounded on associative memory."""

from __future__ import annotations

import logging
import time

from lqnn.core.associative_memory import AssociativeMemory, CollapseResult
from lqnn.models.llm_engine import LLMEngine
from lqnn.system.training_db import TrainingDB

log = logging.getLogger(__name__)


class ChatEngine:
    """Process user chat turns using the quantum associative memory.

    Flow:
    1. Encode question with CLIP
    2. Query ChromaDB for related concepts (collapse)
    3. If confidence is sufficient, generate answer grounded on context
    4. If confidence is low, admit ignorance and optionally trigger learning
    5. Log everything to MongoDB
    """

    MIN_CONFIDENCE = 0.15

    def __init__(self, memory: AssociativeMemory, llm: LLMEngine,
                 training_db: TrainingDB | None = None) -> None:
        self.memory = memory
        self.llm = llm
        self.training_db = training_db

    def chat(self, user_text: str, history: list[dict] | None = None) -> dict:
        """Process a single chat turn."""
        t0 = time.time()
        user_text = user_text.strip()
        if not user_text:
            return {
                "response": "...",
                "confidence": 0.0,
                "concepts": [],
                "duration_ms": 0,
            }

        collapse = self.memory.query(user_text, n_results=10)

        if collapse.confidence >= self.MIN_CONFIDENCE and collapse.context:
            response = self.llm.answer_with_context(
                user_text, collapse.context, max_new_tokens=400,
            )
        elif collapse.confidence > 0:
            response = self.llm.generate(
                f"You have very limited knowledge about this topic. "
                f"What you know: {collapse.context}\n\n"
                f"Question: {user_text}\n"
                f"Give a brief, honest answer. Say what you don't know.",
                max_new_tokens=300,
            )
        else:
            response = self.llm.generate(
                f"You don't have specific knowledge about this topic yet. "
                f"The user asked: {user_text}\n"
                f"Give a brief honest answer and mention that you're still learning.",
                max_new_tokens=200,
            )

        concepts_found = [
            c.get("document", "") for c in collapse.matched_concepts[:5]
            if c.get("document")
        ]

        duration_ms = int((time.time() - t0) * 1000)

        if self.training_db:
            self.training_db.log_chat_turn(
                user_text=user_text,
                response=response,
                confidence=collapse.confidence,
                context_concepts=concepts_found,
            )

        self._maybe_learn(user_text, collapse)

        return {
            "response": response,
            "confidence": round(collapse.confidence, 3),
            "concepts": concepts_found,
            "associations": len(collapse.associations),
            "duration_ms": duration_ms,
        }

    def _maybe_learn(self, text: str, collapse: CollapseResult) -> None:
        """If the query revealed a knowledge gap, learn from the question itself."""
        if collapse.confidence < 0.2:
            words = text.split()
            if 1 <= len(words) <= 5:
                try:
                    self.memory.learn_concept(text, source="user_query")
                except Exception:
                    pass

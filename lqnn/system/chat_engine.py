"""Chat engine -- answers user questions grounded on associative memory.

Now includes reactive learning: when the brain doesn't know something,
it queues an active search to learn it for next time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

from lqnn.core.associative_memory import AssociativeMemory, CollapseResult
from lqnn.models.llm_engine import LLMEngine
from lqnn.system.training_db import TrainingDB

log = logging.getLogger(__name__)

LLM_LOADING_MSG = (
    "[ QUANTUM BRAIN INITIALIZING ]\n"
    "Neural pathways loading... Please wait."
)

REACTIVE_CONFIDENCE_THRESHOLD = 0.30


class ChatEngine:
    """Process user chat turns using the quantum associative memory.

    Flow:
    1. Encode question with CLIP
    2. Query ChromaDB for related concepts (collapse)
    3. If confidence is sufficient, generate answer grounded on context
    4. If confidence is low, admit ignorance and trigger reactive learning
    5. Log everything to MongoDB
    """

    MIN_CONFIDENCE = 0.15
    GENERATION_TIMEOUT_S = 90

    def __init__(self, memory: AssociativeMemory, llm: LLMEngine,
                 training_db: TrainingDB | None = None) -> None:
        self.memory = memory
        self.llm = llm
        self.training_db = training_db
        self._chat_history: list[dict] = []
        self._learning_queue: asyncio.Queue | None = None
        self._reactive_callback: Callable | None = None
        self._recently_queued: set[str] = set()

    def set_reactive_callback(self, callback: Callable) -> None:
        """Set callback for reactive learning (called with topic string)."""
        self._reactive_callback = callback

    def set_learning_queue(self, queue: asyncio.Queue) -> None:
        """Set async queue for reactive learning topics."""
        self._learning_queue = queue

    @property
    def chat_history(self) -> list[dict]:
        return self._chat_history

    def chat(self, user_text: str, history: list[dict] | None = None) -> dict:
        """Process a single chat turn with full error handling."""
        t0 = time.time()
        user_text = user_text.strip()
        if not user_text:
            return {
                "response": "...",
                "confidence": 0.0,
                "concepts": [],
                "duration_ms": 0,
            }

        if self.llm.loading:
            return {
                "response": LLM_LOADING_MSG,
                "confidence": 0.0,
                "concepts": [],
                "duration_ms": 0,
                "status": "loading",
            }

        try:
            return self._process_chat(user_text, t0)
        except Exception as exc:
            log.error("Chat error: %s", exc, exc_info=True)
            duration_ms = int((time.time() - t0) * 1000)
            error_response = (
                f"[ NEURAL ERROR ]\n"
                f"Brain encountered an issue: {type(exc).__name__}\n"
                f"Retrying may help. The brain is still learning."
            )
            return {
                "response": error_response,
                "confidence": 0.0,
                "concepts": [],
                "duration_ms": duration_ms,
                "status": "error",
            }

    def _process_chat(self, user_text: str, t0: float) -> dict:
        collapse = self.memory.query(user_text, n_results=10)

        learning_triggered = False

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

        if collapse.confidence < REACTIVE_CONFIDENCE_THRESHOLD:
            learning_triggered = self._trigger_reactive_learning(user_text)
            if learning_triggered:
                response += (
                    "\n\n[ LEARNING ] I'm actively searching for more "
                    "information about this topic. Ask me again soon!"
                )

        concepts_found = [
            c.get("document", "") for c in collapse.matched_concepts[:5]
            if c.get("document")
        ]

        duration_ms = int((time.time() - t0) * 1000)

        turn = {
            "role": "user",
            "text": user_text,
            "response": response,
            "confidence": round(collapse.confidence, 3),
            "concepts": concepts_found,
            "timestamp": time.time(),
        }
        self._chat_history.append(turn)
        if len(self._chat_history) > 200:
            self._chat_history = self._chat_history[-100:]

        if self.training_db:
            try:
                self.training_db.log_chat_turn(
                    user_text=user_text,
                    response=response,
                    confidence=collapse.confidence,
                    context_concepts=concepts_found,
                )
            except Exception:
                pass

        self._maybe_learn(user_text, collapse)

        return {
            "response": response,
            "confidence": round(collapse.confidence, 3),
            "concepts": concepts_found,
            "associations": len(collapse.associations),
            "duration_ms": duration_ms,
            "status": "ok",
            "learning_triggered": learning_triggered,
        }

    def _trigger_reactive_learning(self, topic: str) -> bool:
        """Queue a topic for active search and learning."""
        topic_key = topic.strip().lower()[:50]

        if topic_key in self._recently_queued:
            return False

        if len(self._recently_queued) > 100:
            self._recently_queued.clear()

        self._recently_queued.add(topic_key)

        if self._learning_queue is not None:
            try:
                self._learning_queue.put_nowait(topic)
                log.info("Reactive learning queued: '%s'", topic[:60])
                return True
            except asyncio.QueueFull:
                log.debug("Reactive learning queue full")
                return False

        if self._reactive_callback is not None:
            try:
                self._reactive_callback(topic)
                log.info("Reactive learning triggered: '%s'", topic[:60])
                return True
            except Exception:
                return False

        return False

    def _maybe_learn(self, text: str, collapse: CollapseResult) -> None:
        """If the query revealed a knowledge gap, learn from the question itself."""
        if collapse.confidence < 0.2:
            words = text.split()
            if 1 <= len(words) <= 5:
                try:
                    self.memory.learn_concept(text, source="user_query")
                except Exception:
                    pass

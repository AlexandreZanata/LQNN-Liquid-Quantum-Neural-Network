"""Chat engine -- answers user questions grounded on associative memory.

v2: Leverages the Probabilistic Wave-Collapse Engine for 12x more context,
multi-hop knowledge discovery, and quantum probability confidence scores.

Includes reactive learning: when the brain doesn't know something,
it queues an active search to learn it for next time.
"""

from __future__ import annotations

import asyncio
import logging
import threading
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
        has_multi_hop = bool(getattr(collapse, "multi_hop_concepts", None))
        max_tokens = 500 if has_multi_hop else 300

        if collapse.confidence >= self.MIN_CONFIDENCE and collapse.context:
            response = self.llm.answer_with_context(
                user_text, collapse.context, max_new_tokens=max_tokens,
            )
        elif collapse.confidence > 0:
            response = self.llm.generate(
                f"Partial knowledge:\n{collapse.context}\n\n"
                f"Question: {user_text}\n\n"
                f"Answer using the above knowledge combined with your general knowledge. "
                f"Be helpful and thorough.",
                max_new_tokens=max_tokens,
                system_prompt=(
                    "You are LQNN, a quantum associative brain. "
                    "Answer helpfully using available knowledge and general reasoning. "
                    "Format with markdown when appropriate."
                ),
            )
        else:
            response = self.llm.generate(
                user_text,
                max_new_tokens=300,
                system_prompt=(
                    "You are LQNN, a quantum associative brain that is always learning. "
                    "You can answer any question using your general intelligence. "
                    "Be thorough and well-structured. "
                    "Format with markdown: **bold**, numbered lists, code blocks."
                ),
            )

        if collapse.confidence < REACTIVE_CONFIDENCE_THRESHOLD:
            learning_triggered = self._trigger_reactive_learning(user_text)
            if learning_triggered:
                response += (
                    "\n\n[ LEARNING ] I'm actively searching for more "
                    "information about this topic. Ask me again soon!"
                )

        concepts_found = [
            c.get("document", "") for c in collapse.matched_concepts[:3]
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

        # Background learn -- don't block the response
        self._bg_maybe_learn(user_text, collapse)

        return {
            "response": response,
            "confidence": round(collapse.confidence, 3),
            "concepts": concepts_found,
            "associations": len(collapse.associations),
            "multi_hop": len(getattr(collapse, "multi_hop_concepts", [])),
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

    def chat_stream(self, user_text: str,
                    on_token: Callable[[str], None] | None = None,
                    on_reasoning: Callable[[str], None] | None = None) -> dict:
        """Streaming chat with real-time reasoning display.

        on_token: called with each generated text chunk
        on_reasoning: called with each reasoning/thinking step so the UI
                      can show what the brain is doing in real time
        """
        t0 = time.time()
        user_text = user_text.strip()
        if not user_text:
            return {"response": "...", "confidence": 0.0, "concepts": [], "duration_ms": 0}

        if self.llm.loading:
            return {"response": LLM_LOADING_MSG, "confidence": 0.0, "concepts": [],
                    "duration_ms": 0, "status": "loading"}

        def _reason(step: str) -> None:
            if on_reasoning:
                on_reasoning(step)

        try:
            _reason("Encoding query with CLIP neural encoder...")

            t_clip = time.time()
            collapse = self.memory.query(user_text, n_results=10)
            clip_ms = int((time.time() - t_clip) * 1000)

            multi_hop_count = len(getattr(collapse, "multi_hop_concepts", []))
            _reason(f"Quantum wave-collapse complete ({clip_ms}ms) — "
                    f"confidence: {collapse.confidence:.2f}, "
                    f"{len(collapse.matched_concepts)} concepts matched, "
                    f"{multi_hop_count} multi-hop links")

            if collapse.matched_concepts:
                top_concepts = [c.get("document", "")[:60]
                                for c in collapse.matched_concepts[:3]
                                if c.get("document")]
                if top_concepts:
                    _reason(f"Top associations: {', '.join(top_concepts)}")

            if collapse.confidence >= self.MIN_CONFIDENCE and collapse.context:
                _reason("High confidence — generating context-grounded response...")
                system_prompt = (
                    "You are LQNN, a quantum associative brain with deep knowledge. "
                    "Answer questions using the provided knowledge context. "
                    "Use your general intelligence to provide a complete, well-structured answer. "
                    "If the context is partial, supplement with your reasoning. "
                    "Format with markdown: use **bold**, numbered lists, code blocks when appropriate."
                )
                prompt = f"Knowledge:\n{collapse.context}\n\nQuestion: {user_text}"
            elif collapse.confidence > 0:
                _reason("Partial knowledge found — combining context with general reasoning...")
                system_prompt = (
                    "You are LQNN, a quantum associative brain. You have some relevant knowledge "
                    "but it may be incomplete. Answer the question using what you know and your "
                    "general intelligence. Be helpful and thorough. "
                    "Format with markdown: use **bold**, numbered lists, code blocks when appropriate."
                )
                prompt = (
                    f"Partial knowledge:\n{collapse.context}\n\n"
                    f"Question: {user_text}\n\n"
                    f"Answer using the above knowledge combined with your general knowledge."
                )
            else:
                _reason("No specific knowledge — using general LLM capabilities...")
                system_prompt = (
                    "You are LQNN, a quantum associative brain that is always learning. "
                    "You don't have specific stored knowledge about this topic yet, "
                    "but you can still help using your general intelligence. "
                    "Give a thorough, well-structured answer. "
                    "Format with markdown: use **bold**, numbered lists, code blocks when appropriate."
                )
                prompt = user_text

            has_multi_hop = bool(getattr(collapse, "multi_hop_concepts", None))
            stream_tokens = 500 if has_multi_hop else 300
            _reason("Generating response tokens...")

            full_response = []
            for chunk in self.llm.generate_stream(
                prompt, max_new_tokens=stream_tokens, temperature=0.4,
                system_prompt=system_prompt,
            ):
                full_response.append(chunk)
                if on_token:
                    on_token(chunk)

            response = "".join(full_response).strip()

            learning_triggered = False
            if collapse.confidence < REACTIVE_CONFIDENCE_THRESHOLD:
                _reason("Low confidence detected — triggering reactive learning...")
                learning_triggered = self._trigger_reactive_learning(user_text)
                if learning_triggered:
                    suffix = ("\n\n[ LEARNING ] I'm actively searching for more "
                              "information about this topic. Ask me again soon!")
                    response += suffix
                    if on_token:
                        on_token(suffix)
                    _reason("Reactive search queued for background learning")

            concepts_found = [
                c.get("document", "") for c in collapse.matched_concepts[:3]
                if c.get("document")
            ]
            duration_ms = int((time.time() - t0) * 1000)
            _reason(f"Response complete in {duration_ms}ms")

            turn = {
                "role": "user", "text": user_text, "response": response,
                "confidence": round(collapse.confidence, 3),
                "concepts": concepts_found, "timestamp": time.time(),
            }
            self._chat_history.append(turn)
            if len(self._chat_history) > 200:
                self._chat_history = self._chat_history[-100:]

            self._bg_maybe_learn(user_text, collapse)

            return {
                "response": response,
                "confidence": round(collapse.confidence, 3),
                "concepts": concepts_found,
                "associations": len(collapse.associations),
                "duration_ms": duration_ms,
                "status": "ok",
                "learning_triggered": learning_triggered,
            }
        except Exception as exc:
            log.error("Chat stream error: %s", exc, exc_info=True)
            return {
                "response": f"[ NEURAL ERROR ] {type(exc).__name__}",
                "confidence": 0.0, "concepts": [],
                "duration_ms": int((time.time() - t0) * 1000),
                "status": "error",
            }

    def _bg_maybe_learn(self, text: str, collapse: CollapseResult) -> None:
        """Fire-and-forget background learning so the response is instant."""
        if collapse.confidence < 0.2:
            words = text.split()
            if 1 <= len(words) <= 5:
                threading.Thread(
                    target=self._maybe_learn_sync,
                    args=(text,),
                    daemon=True,
                ).start()

    def _maybe_learn_sync(self, text: str) -> None:
        try:
            self.memory.learn_concept(text, source="user_query")
        except Exception:
            pass

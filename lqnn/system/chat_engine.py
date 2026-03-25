"""Chat engine v3 -- Quantum Coherence Pipeline.

Replaces the single-shot LLM call with a multi-pass generation strategy
that produces 10x more output while maintaining coherence:

Phase 1 -- Outline:     LLM plans the answer structure (~150 tokens)
Phase 2 -- Sections:    LLM generates each section with focused context (~600 tokens each)
Phase 3 -- Verify:      LLM checks coherence against source context (~100 tokens)

Simple queries (short, high confidence) use a fast single-pass path with
1500 tokens (3-5x current) to avoid unnecessary latency.

v3 also injects conversation history (last 5 turns) into every prompt,
giving the model continuity and preventing hallucinated repetition.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from typing import Callable

from lqnn.core.associative_memory import AssociativeMemory, CollapseResult, _is_coherent
from lqnn.models.llm_engine import CancelCriteria, LLMEngine
from lqnn.system.training_db import TrainingDB

log = logging.getLogger(__name__)

LLM_LOADING_MSG = (
    "[ QUANTUM BRAIN INITIALIZING ]\n"
    "Neural pathways loading... Please wait."
)

REACTIVE_CONFIDENCE_THRESHOLD = 0.08


class _CancelledError(Exception):
    """Internal signal: stream generation was cancelled by the user."""

# Quantum Coherence Pipeline constants (tuned for 7B model on 8GB VRAM)
COHERENCE_MAX_SECTIONS = 5
SECTION_MAX_TOKENS = 800
OUTLINE_MAX_TOKENS = 150
VERIFY_MAX_TOKENS = 100
TOTAL_OUTPUT_BUDGET = 6000

# Single-pass thresholds -- favour fast path (Quantum Tunneling)
SIMPLE_QUERY_MAX_WORDS = 15
SIMPLE_QUERY_MIN_CONFIDENCE = 0.2
SINGLE_PASS_MAX_TOKENS = 1500

# Conversation memory
CONVERSATION_HISTORY_TURNS = 10
CONVERSATION_HISTORY_USER_CHARS = 500
CONVERSATION_HISTORY_ASSISTANT_CHARS = 1000


class ChatEngine:
    """Quantum Coherence Pipeline -- multi-pass generation engine.

    Flow:
    1. Encode question with CLIP + HEI pre-filtered wave-collapse
    2. Superposition context assembly (~50K chars)
    3. Conversation history injection (last 5 turns)
    4. Multi-pass coherent generation (outline -> sections -> verify)
    5. Reactive learning on low confidence
    6. Log everything to MongoDB
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
        self._active_cancel: CancelCriteria | None = None
        self._cancel_lock = threading.Lock()

    def cancel_active(self) -> bool:
        """Cancel the currently running stream generation, if any."""
        with self._cancel_lock:
            if self._active_cancel is not None:
                self._active_cancel.cancel()
                log.info("ChatEngine: active generation cancelled")
                return True
        return False

    def set_reactive_callback(self, callback: Callable) -> None:
        self._reactive_callback = callback

    def set_learning_queue(self, queue: asyncio.Queue) -> None:
        self._learning_queue = queue

    @property
    def chat_history(self) -> list[dict]:
        return self._chat_history

    # ------------------------------------------------------------------ #
    # Conversation memory                                                  #
    # ------------------------------------------------------------------ #

    def _build_conversation_prefix(self) -> str:
        """Build a compact conversation history string from recent turns."""
        recent = self._chat_history[-CONVERSATION_HISTORY_TURNS:]
        if not recent:
            return ""
        parts = []
        for turn in recent:
            user = turn.get("text", "")[:CONVERSATION_HISTORY_USER_CHARS]
            assistant = turn.get("response", "")[:CONVERSATION_HISTORY_ASSISTANT_CHARS]
            if user:
                parts.append(f"User: {user}")
            if assistant:
                parts.append(f"Assistant: {assistant}")
        if not parts:
            return ""
        return "Previous conversation:\n" + "\n".join(parts) + "\n\n"

    # ------------------------------------------------------------------ #
    # Query complexity classification                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_simple_query(user_text: str, confidence: float,
                         multi_hop_count: int) -> bool:
        """Decide whether to use fast single-pass or full coherence pipeline."""
        words = user_text.split()
        if len(words) <= SIMPLE_QUERY_MAX_WORDS and confidence >= SIMPLE_QUERY_MIN_CONFIDENCE:
            return True
        if multi_hop_count == 0 and confidence >= 0.6:
            return True
        return False

    # ------------------------------------------------------------------ #
    # Quantum Coherence Pipeline -- multi-pass generation                  #
    # ------------------------------------------------------------------ #

    def _coherence_pipeline(self, user_text: str, context: str,
                            system_base: str, conversation_prefix: str,
                            ) -> str:
        """Multi-pass generation: outline -> sections -> verify.

        Produces up to ~3000 tokens of coherent, verified output.
        """
        outline_prompt = (
            f"{conversation_prefix}"
            f"Knowledge:\n{context[:6000]}\n\n"
            f"Question: {user_text}\n\n"
            f"Create a brief structured outline (3-5 sections) for a comprehensive "
            f"answer. Output ONLY the outline as a numbered list of section titles. "
            f"No content yet, just the plan."
        )
        outline = self.llm.generate(
            outline_prompt,
            max_new_tokens=OUTLINE_MAX_TOKENS,
            temperature=0.3,
            system_prompt=system_base,
        ).strip()

        sections = self._parse_outline_sections(outline)
        if not sections or len(sections) < 2:
            return self._single_pass_generate(
                user_text, context, system_base, conversation_prefix)

        sections = sections[:COHERENCE_MAX_SECTIONS]

        generated_sections: list[str] = []
        tokens_used = 0

        for i, section_title in enumerate(sections):
            if tokens_used >= TOTAL_OUTPUT_BUDGET:
                break
            remaining = TOTAL_OUTPUT_BUDGET - tokens_used
            section_tokens = min(SECTION_MAX_TOKENS, remaining)
            if section_tokens < 50:
                break

            section_prompt = (
                f"{conversation_prefix}"
                f"Knowledge:\n{context[:8000]}\n\n"
                f"Question: {user_text}\n\n"
                f"You are writing section {i + 1}/{len(sections)} of a comprehensive answer.\n"
                f"Section title: {section_title}\n"
            )
            if generated_sections:
                section_prompt += (
                    f"Previous sections already written:\n"
                    f"{''.join(generated_sections[-3:])}\n\n"
                    f"Continue with this section. Do NOT repeat information "
                    f"from previous sections. "
                )
            section_prompt += (
                "Write ONLY this section. Be thorough, detailed, and well-structured. "
                "Use markdown formatting."
            )

            section_text = self.llm.generate(
                section_prompt,
                max_new_tokens=section_tokens,
                temperature=0.4,
                system_prompt=system_base,
            ).strip()

            if section_text:
                generated_sections.append(
                    f"\n### {section_title}\n\n{section_text}\n")
                tokens_used += len(section_text.split())

        if not generated_sections:
            return self._single_pass_generate(
                user_text, context, system_base, conversation_prefix)

        combined = "".join(generated_sections).strip()

        verified = self._verify_coherence(combined, context, system_base)
        return verified

    def _verify_coherence(self, answer: str, context: str,
                          system_base: str) -> str:
        """Iterative per-section quantum coherence verification.

        Checks each section individually against source context.
        Only rewrites sections that fail -- preserves verified content.
        Max 2 verification passes to bound latency.
        """
        sections = re.split(r'\n### ', answer)
        if len(sections) <= 1:
            return self._verify_section(answer, context, system_base)

        header = sections[0]
        body_sections = sections[1:]

        verified_parts = [header] if header.strip() else []

        for section_text in body_sections:
            section_with_header = f"### {section_text}"
            truncated_section = section_with_header[:3000]

            verify_prompt = (
                f"Context (source of truth):\n{context[:6000]}\n\n"
                f"Section to verify:\n{truncated_section}\n\n"
                f"Does this section contradict the context? Reply with:\n"
                f"- COHERENT (if consistent)\n"
                f"- CORRECTION: <brief description of what to fix>\n"
                f"Reply with ONLY one of the above."
            )
            verdict = self.llm.generate(
                verify_prompt,
                max_new_tokens=VERIFY_MAX_TOKENS,
                temperature=0.1,
                system_prompt=system_base,
            ).strip()

            if "COHERENT" in verdict.upper():
                verified_parts.append(f"\n{section_with_header}")
            else:
                correction_match = re.search(
                    r"CORRECTION:\s*(.+)", verdict,
                    re.IGNORECASE | re.DOTALL)
                if correction_match:
                    correction = correction_match.group(1).strip()
                    fix_prompt = (
                        f"Original section:\n{truncated_section}\n\n"
                        f"Issue: {correction}\n\n"
                        f"Context:\n{context[:6000]}\n\n"
                        f"Rewrite ONLY this section to fix the issue. "
                        f"Keep the same structure and heading."
                    )
                    fixed = self.llm.generate(
                        fix_prompt,
                        max_new_tokens=SECTION_MAX_TOKENS,
                        temperature=0.3,
                        system_prompt=system_base,
                    ).strip()
                    verified_parts.append(
                        f"\n{fixed}" if fixed else f"\n{section_with_header}")
                else:
                    verified_parts.append(f"\n{section_with_header}")

        return "".join(verified_parts).strip()

    def _verify_section(self, text: str, context: str,
                        system_base: str) -> str:
        """Quick single-section coherence check."""
        verify_prompt = (
            f"Context (source of truth):\n{context[:6000]}\n\n"
            f"Generated answer:\n{text[:3000]}\n\n"
            f"Does the answer contradict the context? Reply COHERENT or "
            f"CORRECTION: <what to fix>"
        )
        verdict = self.llm.generate(
            verify_prompt,
            max_new_tokens=VERIFY_MAX_TOKENS,
            temperature=0.1,
            system_prompt=system_base,
        ).strip()

        if "COHERENT" in verdict.upper():
            return text

        correction_match = re.search(
            r"CORRECTION:\s*(.+)", verdict, re.IGNORECASE | re.DOTALL)
        if correction_match:
            correction = correction_match.group(1).strip()
            fix_prompt = (
                f"Original answer:\n{text[:4000]}\n\n"
                f"Issue found: {correction}\n\n"
                f"Rewrite ONLY the problematic parts. Keep everything else."
            )
            fixed = self.llm.generate(
                fix_prompt,
                max_new_tokens=SINGLE_PASS_MAX_TOKENS,
                temperature=0.3,
                system_prompt=system_base,
            ).strip()
            return fixed if fixed else text

        return text

    @staticmethod
    def _parse_outline_sections(outline: str) -> list[str]:
        """Extract section titles from a numbered outline."""
        sections = []
        for line in outline.strip().splitlines():
            cleaned = re.sub(r"^\d+[\.\):\-]\s*", "", line.strip())
            cleaned = re.sub(r"^\*+\s*", "", cleaned).strip()
            if cleaned and len(cleaned) > 3:
                sections.append(cleaned)
        return sections

    def _single_pass_generate(self, user_text: str, context: str,
                              system_base: str, conversation_prefix: str,
                              ) -> str:
        """Fast path for simple queries -- single LLM call, 1500 tokens."""
        prompt = (
            f"{conversation_prefix}"
            f"Knowledge:\n{context}\n\n"
            f"Question: {user_text}\n\n"
            f"Provide a thorough, well-structured answer using the knowledge above. "
            f"Be comprehensive and use markdown formatting."
        )
        return self.llm.generate(
            prompt,
            max_new_tokens=SINGLE_PASS_MAX_TOKENS,
            temperature=0.4,
            system_prompt=system_base,
        ).strip()

    # ------------------------------------------------------------------ #
    # Main chat entry point                                                #
    # ------------------------------------------------------------------ #

    def chat(self, user_text: str, history: list[dict] | None = None) -> dict:
        """Process a single chat turn with the Quantum Coherence Pipeline."""
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
        self.llm.set_chat_active(True)
        try:
            return self._process_chat_impl(user_text, t0)
        finally:
            self.llm.set_chat_active(False)

    def _process_chat_impl(self, user_text: str, t0: float) -> dict:
        collapse = self.memory.query(user_text, n_results=10)

        learning_triggered = False
        multi_hop_count = len(getattr(collapse, "multi_hop_concepts", []))
        conversation_prefix = self._build_conversation_prefix()
        coherence_passes = 0

        system_base = (
            "You are LQNN, a quantum associative brain with deep knowledge. "
            "Answer questions thoroughly using provided context and your reasoning. "
            "Be comprehensive, accurate, and well-structured. "
            "Format with markdown: use **bold**, numbered lists, code blocks when appropriate."
        )

        if collapse.confidence >= self.MIN_CONFIDENCE and collapse.context:
            is_simple = self._is_simple_query(
                user_text, collapse.confidence, multi_hop_count)

            if is_simple:
                response = self._single_pass_generate(
                    user_text, collapse.context, system_base,
                    conversation_prefix)
                coherence_passes = 1
            else:
                response = self._coherence_pipeline(
                    user_text, collapse.context, system_base,
                    conversation_prefix)
                coherence_passes = 3 + min(
                    COHERENCE_MAX_SECTIONS,
                    len(self._parse_outline_sections(response)))

        elif collapse.confidence > 0:
            partial_system = (
                "You are LQNN, a quantum associative brain. You have partial knowledge. "
                "Answer using what you know combined with general reasoning. "
                "Be helpful and thorough. Format with markdown."
            )
            prompt = (
                f"{conversation_prefix}"
                f"Partial knowledge:\n{collapse.context}\n\n"
                f"Question: {user_text}\n\n"
                f"Answer using the above knowledge combined with your general knowledge. "
                f"Be comprehensive and well-structured."
            )
            response = self.llm.generate(
                prompt,
                max_new_tokens=SINGLE_PASS_MAX_TOKENS,
                temperature=0.4,
                system_prompt=partial_system,
            ).strip()
            coherence_passes = 1

        else:
            general_system = (
                "You are LQNN, a quantum associative brain that is always learning. "
                "You can answer any question using your general intelligence. "
                "Be thorough and well-structured. "
                "Format with markdown: **bold**, numbered lists, code blocks."
            )
            prompt = f"{conversation_prefix}{user_text}"
            response = self.llm.generate(
                prompt,
                max_new_tokens=SINGLE_PASS_MAX_TOKENS,
                temperature=0.5,
                system_prompt=general_system,
            ).strip()
            coherence_passes = 1

        if collapse.confidence < REACTIVE_CONFIDENCE_THRESHOLD:
            learning_triggered = self._trigger_reactive_learning(user_text)
            if learning_triggered:
                response += (
                    "\n\n[ LEARNING ] I'm actively searching for more "
                    "information about this topic. Ask me again soon!"
                )

        concepts_found = [
            c.get("document", "") for c in collapse.matched_concepts[:5]
            if c.get("document") and _is_coherent(c["document"])
        ]

        duration_ms = int((time.time() - t0) * 1000)

        ctx_chars = len(collapse.context)
        turn = {
            "role": "user",
            "text": user_text,
            "response": response[:CONVERSATION_HISTORY_ASSISTANT_CHARS],
            "confidence": round(collapse.confidence, 3),
            "concepts": concepts_found,
            "timestamp": time.time(),
            "coherence_passes": coherence_passes,
            "context_chars": ctx_chars,
        }
        self._chat_history.append(turn)
        if len(self._chat_history) > 50:
            self._chat_history = self._chat_history[-30:]

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

        self._bg_maybe_learn(user_text, collapse)

        return {
            "response": response,
            "confidence": round(collapse.confidence, 3),
            "concepts": concepts_found,
            "associations": len(collapse.associations),
            "multi_hop": multi_hop_count,
            "duration_ms": duration_ms,
            "status": "ok",
            "learning_triggered": learning_triggered,
            "coherence_passes": coherence_passes,
            "context_chars": ctx_chars,
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

    # ------------------------------------------------------------------ #
    # Streaming chat with Coherence Pipeline                               #
    # ------------------------------------------------------------------ #

    def chat_stream(self, user_text: str,
                    on_token: Callable[[str], None] | None = None,
                    on_reasoning: Callable[[str], None] | None = None) -> dict:
        """Streaming chat with quantum coherence pipeline.

        For complex queries: streams each section as it generates, with
        reasoning callbacks between sections.
        For simple queries: streams a single-pass 4000-token response.

        Supports mid-stream cancellation via cancel_active().
        """
        cancel = CancelCriteria()
        with self._cancel_lock:
            self._active_cancel = cancel
        self.llm.set_chat_active(True)
        t0 = time.time()
        user_text = user_text.strip()
        if not user_text:
            return {"response": "...", "confidence": 0.0, "concepts": [],
                    "duration_ms": 0}

        if self.llm.loading:
            return {"response": LLM_LOADING_MSG, "confidence": 0.0,
                    "concepts": [], "duration_ms": 0, "status": "loading"}

        def _reason(step: str) -> None:
            if on_reasoning:
                on_reasoning(step)

        try:
            _reason("Encoding query with CLIP + HEI pre-filter...")

            t_clip = time.time()
            collapse = self.memory.query(user_text, n_results=10)
            clip_ms = int((time.time() - t_clip) * 1000)

            multi_hop_count = len(getattr(collapse, "multi_hop_concepts", []))
            _reason(f"Quantum wave-collapse complete ({clip_ms}ms) — "
                    f"confidence: {collapse.confidence:.2f}, "
                    f"{len(collapse.matched_concepts)} concepts matched, "
                    f"{multi_hop_count} multi-hop links, "
                    f"{len(collapse.context)} chars context")

            if collapse.matched_concepts:
                top_concepts = [c.get("document", "")[:60]
                                for c in collapse.matched_concepts[:5]
                                if c.get("document") and _is_coherent(c["document"])]
                if top_concepts:
                    _reason(f"Top associations: {', '.join(top_concepts[:3])}")

            conversation_prefix = self._build_conversation_prefix()
            system_base = (
                "You are LQNN, a quantum associative brain with deep knowledge. "
                "Answer questions thoroughly using provided context and your reasoning. "
                "Be comprehensive, accurate, and well-structured. "
                "Format with markdown: use **bold**, numbered lists, code blocks."
            )

            is_simple = self._is_simple_query(
                user_text, collapse.confidence, multi_hop_count)

            if collapse.confidence >= self.MIN_CONFIDENCE and collapse.context:
                if is_simple:
                    _reason("Simple query — streaming single-pass response "
                            f"(up to {SINGLE_PASS_MAX_TOKENS} tokens)...")
                    prompt = (
                        f"{conversation_prefix}"
                        f"Knowledge:\n{collapse.context}\n\n"
                        f"Question: {user_text}\n\n"
                        f"Provide a thorough, well-structured answer."
                    )
                    full_response = []
                    for chunk in self.llm.generate_stream(
                        prompt, max_new_tokens=SINGLE_PASS_MAX_TOKENS,
                        temperature=0.4, system_prompt=system_base,
                        cancel_criteria=cancel,
                    ):
                        full_response.append(chunk)
                        if on_token:
                            on_token(chunk)
                    response = "".join(full_response).strip()
                else:
                    _reason("Complex query — activating Quantum Coherence Pipeline...")
                    _reason("Phase 1: Generating answer outline...")

                    outline_prompt = (
                        f"{conversation_prefix}"
                        f"Knowledge:\n{collapse.context[:6000]}\n\n"
                        f"Question: {user_text}\n\n"
                        f"Create a brief structured outline (3-5 sections) for "
                        f"a comprehensive answer. Output ONLY the outline as a "
                        f"numbered list of section titles."
                    )
                    outline_chunks = []
                    for chunk in self.llm.generate_stream(
                        outline_prompt, max_new_tokens=OUTLINE_MAX_TOKENS,
                        temperature=0.3, system_prompt=system_base,
                        cancel_criteria=cancel,
                    ):
                        outline_chunks.append(chunk)
                    outline = "".join(outline_chunks).strip()

                    if cancel.is_cancelled:
                        response = ""
                        raise _CancelledError()

                    sections = self._parse_outline_sections(outline)
                    if not sections or len(sections) < 2:
                        _reason("Outline too simple — falling back to single pass...")
                        prompt = (
                            f"{conversation_prefix}"
                            f"Knowledge:\n{collapse.context}\n\n"
                            f"Question: {user_text}\n\n"
                            f"Provide a thorough, well-structured answer."
                        )
                        full_response = []
                        for chunk in self.llm.generate_stream(
                            prompt, max_new_tokens=SINGLE_PASS_MAX_TOKENS,
                            temperature=0.4, system_prompt=system_base,
                            cancel_criteria=cancel,
                        ):
                            full_response.append(chunk)
                            if on_token:
                                on_token(chunk)
                        response = "".join(full_response).strip()
                    else:
                        sections = sections[:COHERENCE_MAX_SECTIONS]
                        _reason(f"Phase 2: Generating {len(sections)} sections...")

                        full_response_parts: list[str] = []
                        tokens_used = 0

                        for i, section_title in enumerate(sections):
                            if cancel.is_cancelled:
                                _reason("Generation cancelled by user")
                                break
                            if tokens_used >= TOTAL_OUTPUT_BUDGET:
                                break
                            remaining = TOTAL_OUTPUT_BUDGET - tokens_used
                            section_tokens = min(SECTION_MAX_TOKENS, remaining)
                            if section_tokens < 50:
                                break

                            _reason(f"  Section {i + 1}/{len(sections)}: "
                                    f"{section_title}")

                            header = f"\n### {section_title}\n\n"
                            if on_token:
                                on_token(header)

                            section_prompt = (
                                f"{conversation_prefix}"
                                f"Knowledge:\n{collapse.context[:8000]}\n\n"
                                f"Question: {user_text}\n\n"
                                f"Section {i + 1}/{len(sections)}: "
                                f"{section_title}\n"
                            )
                            if full_response_parts:
                                section_prompt += (
                                    f"Already written:\n"
                                    f"{''.join(full_response_parts[-3:])}\n\n"
                                    f"Continue. Do NOT repeat. "
                                )
                            section_prompt += (
                                "Write ONLY this section. Be thorough "
                                "and use markdown."
                            )

                            section_chunks: list[str] = []
                            for chunk in self.llm.generate_stream(
                                section_prompt,
                                max_new_tokens=section_tokens,
                                temperature=0.4,
                                system_prompt=system_base,
                                cancel_criteria=cancel,
                            ):
                                section_chunks.append(chunk)
                                if on_token:
                                    on_token(chunk)

                            section_text = "".join(section_chunks).strip()
                            if section_text:
                                full_response_parts.append(
                                    f"\n### {section_title}\n\n"
                                    f"{section_text}\n")
                                tokens_used += len(section_text.split())

                        response = "".join(full_response_parts).strip()

            elif collapse.confidence > 0:
                _reason("Partial knowledge — streaming enhanced response...")
                partial_system = (
                    "You are LQNN, a quantum associative brain with partial knowledge. "
                    "Combine available knowledge with general reasoning. "
                    "Format with markdown."
                )
                prompt = (
                    f"{conversation_prefix}"
                    f"Partial knowledge:\n{collapse.context}\n\n"
                    f"Question: {user_text}\n\n"
                    f"Answer comprehensively."
                )
                full_response = []
                for chunk in self.llm.generate_stream(
                    prompt, max_new_tokens=SINGLE_PASS_MAX_TOKENS,
                    temperature=0.4, system_prompt=partial_system,
                    cancel_criteria=cancel,
                ):
                    full_response.append(chunk)
                    if on_token:
                        on_token(chunk)
                response = "".join(full_response).strip()
            else:
                _reason("No stored knowledge — using general LLM capabilities...")
                general_system = (
                    "You are LQNN, a quantum associative brain always learning. "
                    "Give a thorough, well-structured answer. "
                    "Format with markdown."
                )
                prompt = f"{conversation_prefix}{user_text}"
                full_response = []
                for chunk in self.llm.generate_stream(
                    prompt, max_new_tokens=SINGLE_PASS_MAX_TOKENS,
                    temperature=0.5, system_prompt=general_system,
                    cancel_criteria=cancel,
                ):
                    full_response.append(chunk)
                    if on_token:
                        on_token(chunk)
                response = "".join(full_response).strip()

            learning_triggered = False
            if collapse.confidence < REACTIVE_CONFIDENCE_THRESHOLD:
                _reason("Low confidence — triggering reactive learning...")
                learning_triggered = self._trigger_reactive_learning(user_text)
                if learning_triggered:
                    suffix = ("\n\n[ LEARNING ] I'm actively searching for more "
                              "information about this topic. Ask me again soon!")
                    response += suffix
                    if on_token:
                        on_token(suffix)
                    _reason("Reactive search queued for background learning")

            concepts_found = [
                c.get("document", "") for c in collapse.matched_concepts[:5]
                if c.get("document") and _is_coherent(c["document"])
            ]
            duration_ms = int((time.time() - t0) * 1000)
            _reason(f"Response complete in {duration_ms}ms "
                    f"({len(response)} chars output)")

            stream_ctx_chars = len(collapse.context)
            turn = {
                "role": "user", "text": user_text,
                "response": response[:CONVERSATION_HISTORY_ASSISTANT_CHARS],
                "confidence": round(collapse.confidence, 3),
                "concepts": concepts_found, "timestamp": time.time(),
                "context_chars": stream_ctx_chars,
            }
            self._chat_history.append(turn)
            if len(self._chat_history) > 50:
                self._chat_history = self._chat_history[-30:]

            self._bg_maybe_learn(user_text, collapse)

            return {
                "response": response,
                "confidence": round(collapse.confidence, 3),
                "concepts": concepts_found,
                "associations": len(collapse.associations),
                "duration_ms": duration_ms,
                "status": "ok",
                "learning_triggered": learning_triggered,
                "context_chars": stream_ctx_chars,
            }
        except _CancelledError:
            duration_ms = int((time.time() - t0) * 1000)
            return {
                "response": "",
                "confidence": 0.0, "concepts": [],
                "duration_ms": duration_ms,
                "status": "cancelled",
            }
        except Exception as exc:
            log.error("Chat stream error: %s", exc, exc_info=True)
            return {
                "response": f"[ NEURAL ERROR ] {type(exc).__name__}",
                "confidence": 0.0, "concepts": [],
                "duration_ms": int((time.time() - t0) * 1000),
                "status": "error",
            }
        finally:
            self.llm.set_chat_active(False)
            with self._cancel_lock:
                self._active_cancel = None

    # ------------------------------------------------------------------ #
    # Background learning                                                  #
    # ------------------------------------------------------------------ #

    def _bg_maybe_learn(self, text: str, collapse: CollapseResult) -> None:
        """Fire-and-forget background learning + quality feedback."""
        concept_ids = [c.get("id", "") for c in collapse.matched_concepts[:20]
                       if c.get("id")]
        if concept_ids:
            threading.Thread(
                target=self._feedback_sync,
                args=(concept_ids, collapse.confidence, text),
                daemon=True,
            ).start()

        if collapse.confidence < 0.05 and _is_coherent(text):
            words = text.split()
            if 3 <= len(words) <= 8:
                threading.Thread(
                    target=self._maybe_learn_sync,
                    args=(text,),
                    daemon=True,
                ).start()

    def _feedback_sync(self, concept_ids: list[str],
                       confidence: float, query_text: str = "") -> None:
        try:
            self.memory.feedback_response_quality(concept_ids, confidence)
        except Exception:
            pass
        if query_text and confidence >= 0.5:
            try:
                self.memory.amplify_resonance(
                    query_text, concept_ids, confidence)
            except Exception:
                pass

    def _maybe_learn_sync(self, text: str) -> None:
        try:
            self.memory.learn_concept(text, source="user_query")
        except Exception:
            pass

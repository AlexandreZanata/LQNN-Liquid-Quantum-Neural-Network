"""Qwen2.5-7B inference engine for text generation and association extraction.

v3 additions (Quantum Velocity):
- CancellableStoppingCriteria: abort generation mid-stream
- torch.compile() acceleration (1.5-2x faster after warmup)
- Flash Attention 2 / SDPA support via downloader

v2 additions:
- Batch association generation: 5 concepts in a single LLM call (~5x throughput)
- Cross-pollination: include nearest-neighbour context when generating associations
- Strength tensor support: return semantic + temporal + access + centrality scores
"""

from __future__ import annotations

import logging
import re
import threading

import torch
from transformers import StoppingCriteria

log = logging.getLogger(__name__)

_LOAD_LOCK = threading.Lock()


class CancelCriteria(StoppingCriteria):
    """Thread-safe stopping criteria that can be signalled to abort generation."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self._cancelled


class LLMEngine:
    """Wraps Qwen2.5-7B-Instruct for generation and concept association.

    GPU Quantum Exclusion Principle: when a user chat is active, all
    background inference (associations, self-play, training) is paused
    so the GPU operates at full bandwidth for the user.
    """

    GENERATION_TIMEOUT_S = 120

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._ready = False
        self._loading = False
        self._load_error: str | None = None
        self._chat_active = False

    def set_chat_active(self, active: bool) -> None:
        """Signal that a user chat is in progress (GPU exclusion gate)."""
        self._chat_active = active

    @property
    def chat_active(self) -> bool:
        return self._chat_active

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def loading(self) -> bool:
        return self._loading

    def cache_status(self) -> dict:
        from lqnn.models.downloader import llm_cache_status
        return llm_cache_status()

    def load(self) -> None:
        with _LOAD_LOCK:
            if self._ready:
                return
            self._loading = True
            self._load_error = None
            try:
                from lqnn.models.downloader import ensure_llm_model
                self._model, self._tokenizer = ensure_llm_model()

                self._ready = True
                log.info("LLMEngine: Qwen2.5-7B loaded successfully")
            except Exception as exc:
                self._load_error = str(exc)
                log.error("LLMEngine: failed to load model: %s", exc)
                raise
            finally:
                self._loading = False

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9,
                 system_prompt: str | None = None,
                 _background: bool = False) -> str:
        if _background and self._chat_active:
            return ""
        if not self._ready:
            self.load()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        out = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            repetition_penalty=1.15,
            pad_token_id=pad_id,
        )
        decoded = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        return decoded.strip()

    def generate_stream(self, prompt: str, max_new_tokens: int = 200,
                        temperature: float = 0.4, top_p: float = 0.9,
                        system_prompt: str | None = None,
                        cancel_criteria: CancelCriteria | None = None):
        """Generate tokens with streaming via TextIteratorStreamer.

        Yields text chunks as they are produced. First token arrives fast,
        rest flows progressively.

        If cancel_criteria is provided and .cancel() is called externally,
        the generation thread exits cleanly mid-stream.
        """
        if not self._ready:
            self.load()

        from transformers import TextIteratorStreamer

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True,
        )

        stopping = []
        if cancel_criteria is not None:
            stopping.append(cancel_criteria)

        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            repetition_penalty=1.15,
            pad_token_id=pad_id,
            streamer=streamer,
        )
        if stopping:
            gen_kwargs["stopping_criteria"] = stopping

        thread = threading.Thread(
            target=self._model.generate,
            kwargs=gen_kwargs,
            daemon=True,
        )
        thread.start()

        for chunk in streamer:
            if chunk:
                if cancel_criteria and cancel_criteria.is_cancelled:
                    break
                yield chunk

        thread.join(timeout=120)

    def extract_associations(self, concept: str, n: int = 30) -> list[str]:
        """Generate categorized associations for a concept.

        Categories: visual (color, shape, size), sensory (taste, texture, sound),
        semantic (function, context, category), relational (similar, opposite).
        Respects the GPU exclusion gate (yields to user chat).
        """
        if self._chat_active:
            return []
        system_prompt = (
            "You are a knowledge association engine. You generate rich, "
            "diverse associations for concepts, covering visual properties, "
            "sensory qualities, semantic relationships, and contextual links."
        )
        prompt = (
            f"List exactly {n} words or very short phrases strongly associated "
            f'with the concept "{concept}". Include diverse categories:\n'
            f"- Visual: colors, shapes, sizes\n"
            f"- Sensory: taste, texture, smell, sound\n"
            f"- Semantic: function, category, related concepts\n"
            f"- Relational: similar things, opposites, contexts\n\n"
            f"Output ONLY a numbered list, one item per line, no explanations."
        )
        raw = self.generate(prompt, max_new_tokens=400, temperature=0.8,
                            system_prompt=system_prompt)
        items: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip(" -•*")
            if cleaned and len(cleaned) < 60:
                items.append(cleaned.lower())
        return items[:n]

    def describe_concept(self, concept: str) -> str:
        """Generate a short description of a concept for semantic grounding."""
        prompt = (
            f'In exactly 2 sentences, describe what "{concept}" is. '
            f"Be factual and concise."
        )
        return self.generate(prompt, max_new_tokens=100, temperature=0.3)

    def answer_with_context(self, question: str, context: str,
                            max_new_tokens: int = 300) -> str:
        """Answer a question grounded on retrieved context, supplemented by general knowledge."""
        system_prompt = (
            "You are LQNN, a quantum associative brain with deep knowledge. "
            "Answer questions using the provided knowledge context as your primary source. "
            "If the context is partial, supplement with your general reasoning to give "
            "a complete, helpful answer. Format with markdown: use **bold** for emphasis, "
            "numbered lists for steps, and code blocks for code."
        )
        prompt = (
            f"Knowledge:\n{context}\n\n"
            f"Question: {question}"
        )
        return self.generate(prompt, max_new_tokens=max_new_tokens,
                             temperature=0.4, system_prompt=system_prompt)

    def extract_associations_batch(self, concepts: list[str],
                                   n_per_concept: int = 10,
                                   neighbours: dict[str, list[str]] | None = None,
                                   ) -> dict[str, list[str]]:
        """Generate associations for multiple concepts in ONE LLM call.

        ~5x faster than calling extract_associations() per concept.
        If `neighbours` is provided, includes cross-pollination context.
        Returns {concept: [association_words]}.
        """
        if not concepts:
            return {}
        if self._chat_active:
            return {c: [] for c in concepts}
        if not self._ready:
            self.load()

        concept_lines = []
        for c in concepts[:5]:
            line = f'- "{c}"'
            if neighbours and c in neighbours:
                ctx = ", ".join(neighbours[c][:5])
                line += f" (related to: {ctx})"
            concept_lines.append(line)

        system_prompt = (
            "You are a knowledge association engine. You generate rich, "
            "diverse associations for concepts, covering visual properties, "
            "sensory qualities, semantic relationships, and contextual links."
        )
        prompt = (
            f"For each concept below, list exactly {n_per_concept} "
            f"associated words or very short phrases.\n\n"
            f"Concepts:\n" + "\n".join(concept_lines) + "\n\n"
            "Output format (strictly follow):\n"
            "CONCEPT: word1, word2, word3, ...\n"
            "One line per concept. Only words, no explanations."
        )
        raw = self.generate(prompt, max_new_tokens=600, temperature=0.8,
                            system_prompt=system_prompt)

        result: dict[str, list[str]] = {c: [] for c in concepts}
        for line in raw.splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            header, body = line.split(":", 1)
            header_clean = header.strip().strip('"').strip("- ").lower()
            matched_concept = None
            for c in concepts:
                if c.lower() in header_clean or header_clean in c.lower():
                    matched_concept = c
                    break
            if not matched_concept and concepts:
                for c in concepts:
                    if any(w in header_clean for w in c.lower().split()):
                        matched_concept = c
                        break
            if matched_concept:
                words = [w.strip().strip('"').lower()
                         for w in body.split(",") if w.strip()]
                result[matched_concept].extend(
                    [w for w in words if w and len(w) < 60][:n_per_concept])

        for c in concepts:
            if not result[c]:
                result[c] = self.extract_associations(c, n=n_per_concept)

        return result

    def judge_relevance(self, text: str, concept: str) -> float:
        """Use LLM to judge how relevant a text is to a concept (0.0 to 1.0)."""
        prompt = (
            f'On a scale of 0 to 10, how relevant is the following text to '
            f'the concept "{concept}"? Reply with ONLY a single number.\n\n'
            f'Text: "{text[:500]}"'
        )
        raw = self.generate(prompt, max_new_tokens=5, temperature=0.1)
        try:
            score = float(re.search(r"(\d+(?:\.\d+)?)", raw).group(1))
            return min(1.0, max(0.0, score / 10.0))
        except (AttributeError, ValueError):
            return 0.5

"""Qwen2.5-7B inference engine for text generation and association extraction."""

from __future__ import annotations

import logging
import re
import threading

import torch

log = logging.getLogger(__name__)

_LOAD_LOCK = threading.Lock()


class LLMEngine:
    """Wraps Qwen2.5-7B-Instruct for generation and concept association."""

    GENERATION_TIMEOUT_S = 120

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._ready = False
        self._loading = False
        self._load_error: str | None = None

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
                 system_prompt: str | None = None) -> str:
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
        out = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )
        decoded = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        return decoded.strip()

    def extract_associations(self, concept: str, n: int = 30) -> list[str]:
        """Generate categorized associations for a concept.

        Categories: visual (color, shape, size), sensory (taste, texture, sound),
        semantic (function, context, category), relational (similar, opposite).
        """
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
        """Answer a question grounded on retrieved context."""
        system_prompt = (
            "You are a quantum associative brain. Answer questions using ONLY "
            "the provided knowledge context. If the knowledge is insufficient, "
            "say so honestly. Be concise and accurate."
        )
        prompt = (
            f"Knowledge:\n{context}\n\n"
            f"Question: {question}"
        )
        return self.generate(prompt, max_new_tokens=max_new_tokens,
                             temperature=0.4, system_prompt=system_prompt)

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

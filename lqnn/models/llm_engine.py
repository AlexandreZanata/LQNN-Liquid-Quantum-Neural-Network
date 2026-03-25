"""Phi-3.5-mini inference engine for text generation and association extraction."""

from __future__ import annotations

import logging
import re

import torch

log = logging.getLogger(__name__)


class LLMEngine:
    """Wraps Phi-3.5-mini-instruct for generation and concept association."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        from lqnn.models.downloader import ensure_llm_model

        self._model, self._tokenizer = ensure_llm_model()
        self._ready = True

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        if not self._ready:
            self.load()

        messages = [{"role": "user", "content": prompt}]
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

    def extract_associations(self, concept: str, n: int = 20) -> list[str]:
        """Ask the LLM to list words/concepts associated with *concept*.

        Returns a flat list of single-word or short-phrase associations.
        """
        prompt = (
            f"List exactly {n} words or very short phrases strongly associated "
            f"with the concept \"{concept}\". "
            f"Output ONLY a numbered list, one item per line, no explanations."
        )
        raw = self.generate(prompt, max_new_tokens=300, temperature=0.8)
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
            f"In exactly 2 sentences, describe what \"{concept}\" is. "
            f"Be factual and concise."
        )
        return self.generate(prompt, max_new_tokens=100, temperature=0.3)

    def answer_with_context(self, question: str, context: str,
                            max_new_tokens: int = 300) -> str:
        """Answer a question grounded on retrieved context."""
        prompt = (
            f"Based ONLY on the following knowledge, answer the question. "
            f"If the knowledge is insufficient, say 'I don't know yet'.\n\n"
            f"Knowledge:\n{context}\n\n"
            f"Question: {question}"
        )
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.4)

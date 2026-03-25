"""Qwen2.5-7B inference engine for text generation and association extraction.

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

    def generate_stream(self, prompt: str, max_new_tokens: int = 200,
                        temperature: float = 0.4, top_p: float = 0.9,
                        system_prompt: str | None = None):
        """Generate tokens with streaming via TextIteratorStreamer.

        Yields text chunks as they are produced. First token arrives fast,
        rest flows progressively.
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

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            streamer=streamer,
        )

        thread = threading.Thread(
            target=self._model.generate,
            kwargs=gen_kwargs,
            daemon=True,
        )
        thread.start()

        for chunk in streamer:
            if chunk:
                yield chunk

        thread.join(timeout=120)

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
            f"Output format (strictly follow):\n"
            f"CONCEPT: word1, word2, word3, ...\n"
            f"One line per concept. Only words, no explanations."
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

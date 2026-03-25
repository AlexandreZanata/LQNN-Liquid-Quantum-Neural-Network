"""Shared fixtures for LQNN v3 tests.

All tests run WITHOUT GPU and WITHOUT downloading real AI models.
We mock CLIP and LLM at the fixture level so tests are fast and CI-friendly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture()
def tmp_chroma_dir(tmp_path):
    """Provide a temporary ChromaDB directory that is cleaned up after test."""
    d = tmp_path / "chroma_test"
    d.mkdir()
    yield str(d)


@pytest.fixture()
def mock_clip():
    """Return a mock CLIPEncoder that produces deterministic random vectors."""
    clip = MagicMock()
    clip.ready = True
    clip.embed_dim = 512

    def _encode_text(text):
        rng = np.random.RandomState(hash(text) % 2**31)
        v = rng.randn(512).astype(np.float32)
        return v / np.linalg.norm(v)

    def _encode_texts(texts):
        return np.array([_encode_text(t) for t in texts])

    def _encode_image(img):
        rng = np.random.RandomState(42)
        v = rng.randn(512).astype(np.float32)
        return v / np.linalg.norm(v)

    clip.encode_text = MagicMock(side_effect=_encode_text)
    clip.encode_texts = MagicMock(side_effect=_encode_texts)
    clip.encode_image = MagicMock(side_effect=_encode_image)
    clip.load = MagicMock()
    return clip


@pytest.fixture()
def mock_llm():
    """Return a mock LLMEngine that produces canned responses."""
    llm = MagicMock()
    llm.ready = True
    llm.loading = False
    llm.chat_active = False  # avoid MagicMock truthy default in self_play / gating

    def _generate(prompt, **kwargs):
        return "This is a test response from the mock LLM."

    def _extract_associations(concept, n=30):
        base = ["yellow", "sweet", "fruit", "tropical", "monkey",
                "potassium", "peel", "bunch", "ripe", "organic",
                "smoothie", "healthy", "curved", "plantain", "dessert",
                "snack", "energy", "vitamin", "fiber", "natural",
                "soft", "creamy", "elongated", "green", "brown",
                "breakfast", "lunch", "recipe", "garden", "market"]
        return base[:n]

    def _describe_concept(concept):
        return f"{concept} is a common concept. It has many associations."

    def _answer_with_context(question, context, **kwargs):
        return f"Based on the knowledge: {context[:50]}... the answer is related."

    def _judge_relevance(text, concept):
        return 0.7

    llm.generate = MagicMock(side_effect=_generate)
    llm.extract_associations = MagicMock(side_effect=_extract_associations)
    llm.describe_concept = MagicMock(side_effect=_describe_concept)
    llm.answer_with_context = MagicMock(side_effect=_answer_with_context)
    llm.judge_relevance = MagicMock(side_effect=_judge_relevance)
    llm.load = MagicMock()
    return llm


@pytest.fixture()
def vector_store(tmp_chroma_dir):
    """Provide a clean VectorStore backed by a temp directory."""
    from lqnn.core.vector_store import VectorStore
    return VectorStore(persist_dir=tmp_chroma_dir)


@pytest.fixture()
def memory(vector_store, mock_clip, mock_llm):
    """Provide an AssociativeMemory wired to mocked models and temp store."""
    from lqnn.core.associative_memory import AssociativeMemory
    return AssociativeMemory(store=vector_store, clip=mock_clip, llm=mock_llm)

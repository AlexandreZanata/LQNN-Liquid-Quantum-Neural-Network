"""Tests for lqnn.models -- CLIP encoder and LLM engine (mock-level only)."""

import numpy as np
import pytest

from lqnn.models.clip_encoder import CLIPEncoder
from lqnn.models.llm_engine import LLMEngine


class TestCLIPEncoder:

    def test_initial_state(self):
        clip = CLIPEncoder()
        assert clip.ready is False
        assert clip.embed_dim == 512

    def test_similarity_function(self):
        clip = CLIPEncoder()
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        assert clip.similarity(a, b) == pytest.approx(1.0)
        assert clip.similarity(a, c) == pytest.approx(0.0)


class TestLLMEngine:

    def test_initial_state(self):
        llm = LLMEngine()
        assert llm.ready is False

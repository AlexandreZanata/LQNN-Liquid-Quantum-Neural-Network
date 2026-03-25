"""Tests for lqnn.agents -- browser agent and manager."""


import numpy as np

from lqnn.agents.browser_agent import BrowserAgent
from lqnn.agents.manager import AgentManager, JudgeAgent, KnowledgeGap


class TestJudgeAgent:

    def setup_method(self):
        self.judge = JudgeAgent()

    def test_reject_empty_text(self):
        ok, reason = self.judge.judge_text("")
        assert not ok
        assert reason == "text_too_short"

    def test_reject_short_text(self):
        ok, reason = self.judge.judge_text("hi")
        assert not ok
        assert reason == "text_too_short"

    def test_accept_valid_text(self):
        ok, reason = self.judge.judge_text("This is a valid text about bananas and fruits.")
        assert ok
        assert reason == "ok"

    def test_reject_gibberish(self):
        ok, reason = self.judge.judge_text("123456789!@#$%^&*()_+" * 5)
        assert not ok
        assert reason == "text_not_coherent"

    def test_reject_small_image(self):
        ok, reason = self.judge.judge_image(b"\x00" * 10)
        assert not ok
        assert reason == "image_too_small"

    def test_reject_invalid_image(self):
        ok, reason = self.judge.judge_image(b"\x00" * 2000)
        assert not ok
        assert reason == "image_invalid"

    def test_is_duplicate_with_no_concepts(self, memory):
        vec = np.random.randn(512).astype(np.float32)
        assert not self.judge.is_duplicate(memory, vec)


class TestBrowserAgent:

    def test_extract_ddg_url_valid(self):
        url = BrowserAgent._extract_ddg_url(
            "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&rut=abc"
        )
        assert url == "https://example.com"

    def test_extract_ddg_url_direct(self):
        url = BrowserAgent._extract_ddg_url("https://example.com/page")
        assert url == "https://example.com/page"

    def test_extract_ddg_url_none(self):
        url = BrowserAgent._extract_ddg_url("/relative/path")
        assert url is None

    def test_stats_initial(self):
        agent = BrowserAgent()
        stats = agent.stats()
        assert stats["fetch_count"] == 0


class TestAgentManager:

    def test_detect_gaps_empty_memory(self, memory):
        from lqnn.training.continuous_trainer import TrainingPhase
        manager = AgentManager(memory=memory)
        manager.set_phase(TrainingPhase.VISUAL_OBJECTS)
        gaps = manager.detect_knowledge_gaps()

        assert len(gaps) > 0
        assert isinstance(gaps[0], KnowledgeGap)

    def test_detect_gaps_seeded_memory(self, memory):
        from lqnn.training.continuous_trainer import TrainingPhase
        memory.learn_concept("technology", source="test")
        manager = AgentManager(memory=memory)
        manager.set_phase(TrainingPhase.ABSTRACT_CONCEPTS)
        gaps = manager.detect_knowledge_gaps()

        assert isinstance(gaps, list)

    def test_stats_initial(self, memory):
        manager = AgentManager(memory=memory)
        stats = manager.stats()

        assert stats["cycle"] == 0
        assert stats["online"] is True
        assert stats["last_report"] is None

    def test_set_online(self, memory):
        manager = AgentManager(memory=memory)
        manager.set_online(False)
        assert not manager._online
        manager.set_online(True)
        assert manager._online

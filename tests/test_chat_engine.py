"""Tests for lqnn.system.chat_engine."""

import time

from lqnn.system.chat_engine import ChatEngine


class TestChatEngine:

    def test_empty_input(self, memory, mock_llm):
        engine = ChatEngine(memory=memory, llm=mock_llm)
        result = engine.chat("")

        assert result["response"] == "..."
        assert result["confidence"] == 0.0

    def test_chat_with_knowledge(self, memory, mock_llm):
        memory.learn_concept("banana", source="test")
        engine = ChatEngine(memory=memory, llm=mock_llm)

        result = engine.chat("tell me about banana")

        assert "response" in result
        assert "confidence" in result
        assert "concepts" in result
        assert "duration_ms" in result
        assert isinstance(result["confidence"], float)
        assert result["response"] != "..."

    def test_chat_without_knowledge(self, memory, mock_llm):
        engine = ChatEngine(memory=memory, llm=mock_llm)

        result = engine.chat("what is quantum physics?")

        assert "response" in result
        assert result["confidence"] == 0.0

    def test_chat_with_training_db(self, memory, mock_llm):
        from unittest.mock import MagicMock
        mock_db = MagicMock()
        mock_db.log_chat_turn = MagicMock()

        engine = ChatEngine(memory=memory, llm=mock_llm, training_db=mock_db)
        engine.chat("hello")

        mock_db.log_chat_turn.assert_called_once()

    def test_maybe_learn_from_short_query(self, memory, mock_llm):
        engine = ChatEngine(memory=memory, llm=mock_llm)
        initial_count = memory.store.concept_count()

        engine.chat("what is a banana fruit")
        time.sleep(0.5)  # background learning thread

        assert memory.store.concept_count() > initial_count

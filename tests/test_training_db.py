"""Tests for lqnn.system.training_db -- MongoDB integration (mocked)."""

from unittest.mock import patch


from lqnn.system.training_db import TrainingDB


class TestTrainingDB:

    def test_unavailable_gracefully(self):
        with patch("lqnn.system.training_db.TrainingDB._connect"):
            db = TrainingDB.__new__(TrainingDB)
            db._available = False
            db._client = None
            db._db = None

            db.log_chat_turn("hello", "hi", 0.5)
            db.log_training_cycle({"cycle": 1})
            db.log_agent_cycle({"cycle": 1})

            assert db.get_recent_training() == []
            assert db.get_recent_chats() == []

    def test_available_property(self):
        db = TrainingDB.__new__(TrainingDB)
        db._available = True
        assert db.available is True

        db._available = False
        assert db.available is False

"""Tests for UI layer -- renderer and controls."""


from ui.renderer import build_brain_payload


class TestRenderer:

    def test_build_empty_payload(self):
        payload = build_brain_payload()

        assert payload["type"] == "state"
        assert payload["memory"] == {}
        assert payload["training"] == {}
        assert payload["agents"] == {}
        assert payload["chat_history"] == []
        assert payload["recent_concepts"] == []
        assert payload["training_log"] == []
        assert payload["agent_activity"] == []
        assert payload["system"] == {}

    def test_build_payload_with_data(self):
        payload = build_brain_payload(
            memory_stats={"concepts": 10, "associations": 50},
            training_status={"running": True, "cycle": 5},
            chat_history=[{"role": "user", "text": "hello"}],
            recent_concepts=[{"concept": "banana", "volatility": 0.3}],
            training_log=[{"type": "cycle_end", "cycle": 5}],
            agent_activity=[{"type": "learn", "concept": "banana"}],
            system_metrics={"cpu_percent": 45},
        )

        assert payload["memory"]["concepts"] == 10
        assert payload["training"]["running"] is True
        assert len(payload["chat_history"]) == 1
        assert payload["recent_concepts"][0]["concept"] == "banana"
        assert len(payload["training_log"]) == 1
        assert len(payload["agent_activity"]) == 1
        assert payload["system"]["cpu_percent"] == 45

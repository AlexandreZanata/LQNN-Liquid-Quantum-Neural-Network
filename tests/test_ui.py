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

    def test_build_payload_with_data(self):
        payload = build_brain_payload(
            memory_stats={"concepts": 10, "associations": 50},
            training_status={"running": True, "cycle": 5},
            chat_history=[{"role": "user", "text": "hello"}],
            recent_concepts=[{"concept": "banana", "volatility": 0.3}],
        )

        assert payload["memory"]["concepts"] == 10
        assert payload["training"]["running"] is True
        assert len(payload["chat_history"]) == 1
        assert payload["recent_concepts"][0]["concept"] == "banana"

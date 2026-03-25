"""Tests for lqnn.core.associative_memory -- the quantum brain."""


from lqnn.core.associative_memory import AssociativeMemory, CollapseResult, QuantumState


class TestAssociativeMemory:

    def test_learn_concept_text_only(self, memory: AssociativeMemory):
        state = memory.learn_concept("banana", source="test")

        assert isinstance(state, QuantumState)
        assert state.concept == "banana"
        assert state.volatility == 1.0
        assert len(state.associations) > 0
        assert memory.store.concept_count() == 1
        assert memory._learn_count == 1

    def test_learn_concept_with_image(self, memory: AssociativeMemory):
        fake_image = b"\x00" * 1000
        state = memory.learn_concept("apple", image=fake_image, source="test")

        assert state.concept == "apple"
        assert memory.store.concept_count() == 1
        memory.clip.encode_image.assert_called_once()

    def test_learn_multiple_concepts(self, memory: AssociativeMemory):
        memory.learn_concept("banana", source="test")
        memory.learn_concept("apple", source="test")
        memory.learn_concept("car", source="test")

        assert memory.store.concept_count() == 3
        assert memory._learn_count == 3

    def test_query_returns_collapse_result(self, memory: AssociativeMemory):
        memory.learn_concept("banana", source="test")

        result = memory.query("what is a banana?")

        assert isinstance(result, CollapseResult)
        assert result.query == "what is a banana?"
        assert result.confidence >= 0.0
        assert memory._query_count == 1

    def test_query_empty_memory(self, memory: AssociativeMemory):
        result = memory.query("anything")

        assert isinstance(result, CollapseResult)
        assert result.confidence == 0.0
        assert len(result.matched_concepts) == 0

    def test_consolidation_cycle(self, memory: AssociativeMemory):
        memory.learn_concept("banana", source="test")
        memory.learn_concept("apple", source="test")

        result = memory.consolidate()

        assert "pruned" in result
        assert "crystallized" in result
        assert "decayed" in result

    def test_self_play_with_concepts(self, memory: AssociativeMemory):
        memory.learn_concept("banana", source="test")

        result = memory.self_play_cycle()

        assert "action" in result
        assert result["action"] in ("validated", "reinforced", "skip")

    def test_self_play_empty_memory(self, memory: AssociativeMemory):
        result = memory.self_play_cycle()

        assert result["action"] == "skip"
        assert result["reason"] == "no_concepts"

    def test_stats(self, memory: AssociativeMemory):
        stats = memory.stats()

        assert "concepts" in stats
        assert "associations" in stats
        assert "learn_count" in stats
        assert "query_count" in stats
        assert "clip_ready" in stats
        assert "llm_ready" in stats

    def test_make_id_deterministic(self):
        id1 = AssociativeMemory._make_id("banana")
        id2 = AssociativeMemory._make_id("banana")
        id3 = AssociativeMemory._make_id("apple")

        assert id1 == id2
        assert id1 != id3

    def test_duplicate_concept_upserts(self, memory: AssociativeMemory):
        memory.learn_concept("banana", source="test")
        memory.learn_concept("banana", source="test_again")

        assert memory.store.concept_count() == 1

"""Tests for lqnn.core.vector_store -- ChromaDB wrapper."""


import numpy as np

from lqnn.core.vector_store import VectorEntry, VectorStore


class TestVectorStore:

    def test_add_and_query_concept(self, vector_store: VectorStore):
        vec = np.random.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)

        entry = VectorEntry(
            id="test_banana",
            vector=vec,
            concept="banana",
            source="test",
            volatility=0.8,
            confidence=0.7,
        )
        vector_store.add_concept(entry)

        assert vector_store.concept_count() == 1

        results = vector_store.query_concepts(vec, n=5)
        assert len(results) >= 1
        assert results[0]["id"] == "test_banana"
        assert results[0]["document"] == "banana"

    def test_get_concept_by_id(self, vector_store: VectorStore):
        vec = np.random.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)

        entry = VectorEntry(id="c1", vector=vec, concept="apple")
        vector_store.add_concept(entry)

        result = vector_store.get_concept("c1")
        assert result is not None
        assert result["document"] == "apple"

    def test_get_nonexistent_concept(self, vector_store: VectorStore):
        result = vector_store.get_concept("does_not_exist")
        assert result is None

    def test_update_metadata(self, vector_store: VectorStore):
        vec = np.random.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)

        entry = VectorEntry(
            id="up1", vector=vec, concept="cat",
            volatility=0.9, access_count=0,
        )
        vector_store.add_concept(entry)

        vector_store.update_metadata("up1", {"access_count": 5, "volatility": 0.3})

        result = vector_store.get_concept("up1")
        assert result["metadata"]["access_count"] == 5
        assert result["metadata"]["volatility"] == 0.3

    def test_delete_concept(self, vector_store: VectorStore):
        vec = np.random.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)

        entry = VectorEntry(id="del1", vector=vec, concept="dog")
        vector_store.add_concept(entry)
        assert vector_store.concept_count() == 1

        vector_store.delete_concept("del1")
        assert vector_store.concept_count() == 0

    def test_add_and_query_association(self, vector_store: VectorStore):
        vec = np.random.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)

        aid = vector_store.add_association(
            source_concept="banana",
            target_concept="yellow",
            vector=vec,
            strength=0.85,
        )
        assert aid.startswith("assoc_banana_yellow")
        assert vector_store.association_count() == 1

        results = vector_store.query_associations(vec, n=5)
        assert len(results) >= 1

    def test_volatile_and_stable_concepts(self, vector_store: VectorStore):
        for i, vol in enumerate([0.1, 0.5, 0.95]):
            vec = np.random.randn(512).astype(np.float32)
            vec /= np.linalg.norm(vec)
            entry = VectorEntry(
                id=f"vol_{i}", vector=vec, concept=f"concept_{i}",
                volatility=vol,
            )
            vector_store.add_concept(entry)

        volatile = vector_store.get_volatile_concepts(threshold=0.9)
        assert len(volatile) == 1
        assert volatile[0]["id"] == "vol_2"

        stable = vector_store.get_stable_concepts(threshold=0.2)
        assert len(stable) == 1
        assert stable[0]["id"] == "vol_0"

    def test_stats(self, vector_store: VectorStore):
        stats = vector_store.stats()
        assert "concepts" in stats
        assert "associations" in stats
        assert stats["concepts"] == 0
        assert stats["associations"] == 0

"""ChromaDB wrapper for persistent vector storage with quantum metadata."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import chromadb
import numpy as np

log = logging.getLogger(__name__)

CHROMA_DIR = os.environ.get("CHROMA_DIR", "data/chroma")


@dataclass
class VectorEntry:
    id: str
    vector: np.ndarray
    concept: str
    source: str = "unknown"
    volatility: float = 1.0
    confidence: float = 0.5
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """Persistent vector store backed by ChromaDB.

    Collections:
    - concepts: primary concept vectors (CLIP embeddings)
    - associations: relational links between concepts
    """

    def __init__(self, persist_dir: str | None = None) -> None:
        path = persist_dir or CHROMA_DIR
        self._client = chromadb.PersistentClient(path=path)
        self._concepts = self._client.get_or_create_collection(
            name="concepts",
            metadata={"hnsw:space": "cosine"},
        )
        self._associations = self._client.get_or_create_collection(
            name="associations",
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "VectorStore ready: %d concepts, %d associations",
            self._concepts.count(), self._associations.count(),
        )

    # -- concepts --

    def add_concept(self, entry: VectorEntry) -> None:
        meta = {
            "concept": entry.concept,
            "source": entry.source,
            "volatility": entry.volatility,
            "confidence": entry.confidence,
            "access_count": entry.access_count,
            "created_at": entry.created_at,
            "last_accessed": entry.last_accessed,
        }
        meta.update({k: v for k, v in entry.metadata.items()
                     if isinstance(v, (str, int, float, bool))})

        self._concepts.upsert(
            ids=[entry.id],
            embeddings=[entry.vector.tolist()],
            metadatas=[meta],
            documents=[entry.concept],
        )

    def query_concepts(self, vector: np.ndarray, n: int = 10,
                       max_volatility: float | None = None) -> list[dict]:
        """Find nearest concepts. Optionally filter by volatility ceiling."""
        where = None
        if max_volatility is not None:
            where = {"volatility": {"$lte": max_volatility}}

        results = self._concepts.query(
            query_embeddings=[vector.tolist()],
            n_results=min(n, max(self._concepts.count(), 1)),
            where=where,
            include=["metadatas", "documents", "distances", "embeddings"],
        )
        return self._unpack(results)

    def get_concept(self, concept_id: str) -> dict | None:
        try:
            result = self._concepts.get(
                ids=[concept_id],
                include=["metadatas", "documents", "embeddings"],
            )
            ids = result.get("ids", [])
            if not ids:
                return None
            metas = result.get("metadatas")
            docs = result.get("documents")
            embeds = result.get("embeddings")
            return {
                "id": ids[0],
                "metadata": metas[0] if metas is not None and len(metas) > 0 else {},
                "document": docs[0] if docs is not None and len(docs) > 0 else "",
                "embedding": embeds[0] if embeds is not None and len(embeds) > 0 else None,
            }
        except Exception:
            pass
        return None

    def update_metadata(self, concept_id: str, updates: dict) -> None:
        existing = self.get_concept(concept_id)
        if not existing:
            return
        meta = existing["metadata"]
        meta.update(updates)
        self._concepts.update(ids=[concept_id], metadatas=[meta])

    def delete_concept(self, concept_id: str) -> None:
        try:
            self._concepts.delete(ids=[concept_id])
        except Exception:
            pass

    # -- associations --

    def add_association(self, source_concept: str, target_concept: str,
                        vector: np.ndarray, strength: float = 1.0,
                        assoc_id: str | None = None) -> str:
        aid = assoc_id or f"assoc_{source_concept}_{target_concept}_{time.time():.0f}"
        self._associations.upsert(
            ids=[aid],
            embeddings=[vector.tolist()],
            metadatas=[{
                "source_concept": source_concept,
                "target_concept": target_concept,
                "strength": strength,
                "created_at": time.time(),
                "access_count": 0,
            }],
            documents=[f"{source_concept} -> {target_concept}"],
        )
        return aid

    def query_associations(self, vector: np.ndarray, n: int = 20) -> list[dict]:
        count = self._associations.count()
        if count == 0:
            return []
        results = self._associations.query(
            query_embeddings=[vector.tolist()],
            n_results=min(n, count),
            include=["metadatas", "documents", "distances"],
        )
        return self._unpack(results)

    # -- maintenance --

    def concept_count(self) -> int:
        return self._concepts.count()

    def association_count(self) -> int:
        return self._associations.count()

    def get_volatile_concepts(self, threshold: float = 0.9) -> list[dict]:
        """Return concepts with volatility above threshold (candidates for pruning)."""
        results = self._concepts.get(
            where={"volatility": {"$gte": threshold}},
            include=["metadatas", "documents"],
        )
        out = []
        for i, cid in enumerate(results["ids"]):
            out.append({
                "id": cid,
                "metadata": results["metadatas"][i] if results["metadatas"] else {},
                "document": results["documents"][i] if results["documents"] else "",
            })
        return out

    def get_stable_concepts(self, threshold: float = 0.3) -> list[dict]:
        """Return crystallized concepts (low volatility)."""
        results = self._concepts.get(
            where={"volatility": {"$lte": threshold}},
            include=["metadatas", "documents"],
        )
        out = []
        for i, cid in enumerate(results["ids"]):
            out.append({
                "id": cid,
                "metadata": results["metadatas"][i] if results["metadatas"] else {},
                "document": results["documents"][i] if results["documents"] else "",
            })
        return out

    def stats(self) -> dict:
        return {
            "concepts": self._concepts.count(),
            "associations": self._associations.count(),
        }

    @staticmethod
    def _unpack(results: dict) -> list[dict]:
        items = []
        if not results or not results.get("ids"):
            return items
        ids_list = results["ids"][0] if results["ids"] else []
        metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        docs = results.get("documents", [[]])[0] if results.get("documents") else []
        dists = results.get("distances", [[]])[0] if results.get("distances") else []
        embeds = results.get("embeddings", [[]])[0] if results.get("embeddings") else []

        for i, rid in enumerate(ids_list):
            item: dict = {"id": rid}
            if i < len(metas) and metas[i] is not None:
                item["metadata"] = metas[i]
            if i < len(docs) and docs[i] is not None:
                item["document"] = docs[i]
            if i < len(dists):
                item["distance"] = dists[i]
            if i < len(embeds) and embeds[i] is not None:
                item["embedding"] = embeds[i]
            items.append(item)
        return items

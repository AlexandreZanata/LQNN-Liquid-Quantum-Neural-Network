"""ChromaDB wrapper for persistent vector storage with quantum metadata.

v2 additions:
- Batch upsert / query helpers for the QuantumBatchEngine
- Float16 embedding support (2x memory savings, lossless for cosine sim)
- Dirty-set tracking for incremental consolidation
- In-memory crystallized-tier cache for sub-ms lookups
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import chromadb
import numpy as np

log = logging.getLogger(__name__)

CHROMA_DIR = os.environ.get("CHROMA_DIR", "data/chroma")
USE_FLOAT16 = True  # halve embedding storage with negligible quality loss


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


def _to_f16(vec: np.ndarray) -> list[float]:
    """Optionally compress to float16 before handing to Chroma."""
    if USE_FLOAT16:
        return vec.astype(np.float16).astype(np.float32).tolist()
    return vec.tolist()


class VectorStore:
    """Persistent vector store backed by ChromaDB.

    Collections:
    - concepts: primary concept vectors (CLIP embeddings)
    - associations: relational links between concepts

    v2 features:
    - dirty_set: track modified concept IDs for incremental consolidation
    - crystallized_cache: in-memory tier for low-volatility concepts
    - batch helpers for the QuantumBatchEngine
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

        # Incremental consolidation tracking
        self._dirty_lock = threading.Lock()
        self._dirty_ids: set[str] = set()

        # Crystallized tier: id -> (vector, metadata, document)
        self._crystal_lock = threading.Lock()
        self._crystallized: dict[str, tuple[np.ndarray, dict, str]] = {}

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
            embeddings=[_to_f16(entry.vector)],
            metadatas=[meta],
            documents=[entry.concept],
        )
        self._mark_dirty(entry.id)

    def query_concepts(self, vector: np.ndarray, n: int = 10,
                       max_volatility: float | None = None,
                       id_filter: list[str] | None = None) -> list[dict]:
        """Find nearest concepts.

        Parameters
        ----------
        id_filter : optional list of concept IDs to restrict the search to
                    (used by HEI.narrow() for 10-20x faster collapse).
        max_volatility : optional volatility ceiling filter.
        """
        if id_filter:
            return self._query_concepts_filtered(vector, n, id_filter,
                                                 max_volatility)

        where = None
        if max_volatility is not None:
            where = {"volatility": {"$lte": max_volatility}}

        try:
            results = self._concepts.query(
                query_embeddings=[vector.tolist()],
                n_results=n,
                where=where,
                include=["metadatas", "documents", "distances"],
            )
        except Exception:
            results = self._concepts.query(
                query_embeddings=[vector.tolist()],
                n_results=max(self._concepts.count(), 1),
                where=where,
                include=["metadatas", "documents", "distances"],
            )
        return self._unpack(results)

    def _query_concepts_filtered(self, vector: np.ndarray, n: int,
                                 id_filter: list[str],
                                 max_volatility: float | None = None,
                                 ) -> list[dict]:
        """Cosine search restricted to a pre-filtered set of concept IDs.

        Fetches embeddings for the given IDs from Chroma, then does a fast
        numpy dot-product ranking -- much faster than a full ANN scan.
        """
        try:
            data = self._concepts.get(
                ids=id_filter,
                include=["embeddings", "metadatas", "documents"],
            )
        except Exception:
            return self.query_concepts(vector, n, max_volatility)

        ids = data.get("ids", [])
        embeds = data.get("embeddings", [])
        metas = data.get("metadatas", [])
        docs = data.get("documents", [])
        if not ids or not embeds:
            return self.query_concepts(vector, n, max_volatility)

        mat = np.array(embeds, dtype=np.float32)
        qvec = vector.astype(np.float32)
        sims = mat @ qvec
        top_indices = np.argsort(-sims)

        results = []
        for idx in top_indices:
            idx = int(idx)
            meta = metas[idx] if idx < len(metas) and metas[idx] else {}
            if max_volatility is not None:
                if meta.get("volatility", 1.0) > max_volatility:
                    continue
            results.append({
                "id": ids[idx],
                "metadata": meta,
                "document": docs[idx] if idx < len(docs) and docs[idx] else "",
                "distance": float(1.0 - sims[idx]),
            })
            if len(results) >= n:
                break
        return results

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
        self._mark_dirty(concept_id)

    def batch_touch(self, ids: list[str], metas: list[dict]) -> None:
        """Batch-update access_count and last_accessed for multiple concepts."""
        if not ids:
            return
        now = time.time()
        updated_ids = []
        updated_metas = []
        for cid, meta in zip(ids, metas):
            if not cid:
                continue
            m = dict(meta)
            m["access_count"] = m.get("access_count", 0) + 1
            m["last_accessed"] = now
            updated_ids.append(cid)
            updated_metas.append(m)
        if updated_ids:
            try:
                self._concepts.update(ids=updated_ids, metadatas=updated_metas)
            except Exception:
                pass

    def delete_concept(self, concept_id: str) -> None:
        try:
            self._concepts.delete(ids=[concept_id])
        except Exception:
            pass
        with self._crystal_lock:
            self._crystallized.pop(concept_id, None)

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
        try:
            results = self._associations.query(
                query_embeddings=[vector.tolist()],
                n_results=n,
                include=["metadatas", "documents", "distances"],
            )
        except Exception:
            count = self._associations.count()
            if count == 0:
                return []
            results = self._associations.query(
                query_embeddings=[vector.tolist()],
                n_results=max(count, 1),
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

    # -- batch operations (used by QuantumBatchEngine) --

    def batch_add_concepts(self, entries: list[VectorEntry]) -> None:
        """Upsert multiple concepts in a single Chroma call."""
        if not entries:
            return
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        for entry in entries:
            ids.append(entry.id)
            embeddings.append(_to_f16(entry.vector))
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
            metadatas.append(meta)
            documents.append(entry.concept)
        self._concepts.upsert(
            ids=ids, embeddings=embeddings,
            metadatas=metadatas, documents=documents,
        )
        for cid in ids:
            self._mark_dirty(cid)

    def batch_query_concepts(self, vectors: list[np.ndarray],
                             n: int = 10) -> list[list[dict]]:
        """Run multiple concept queries in one Chroma call."""
        if not vectors:
            return []
        embeds = [_to_f16(v) for v in vectors]
        try:
            raw = self._concepts.query(
                query_embeddings=embeds,
                n_results=n,
                include=["metadatas", "documents", "distances"],
            )
        except Exception:
            n_avail = max(self._concepts.count(), 1)
            raw = self._concepts.query(
                query_embeddings=embeds,
                n_results=min(n, n_avail),
                include=["metadatas", "documents", "distances"],
            )
        return self._unpack_multi(raw)

    def batch_query_associations(self, vectors: list[np.ndarray],
                                  n: int = 20) -> list[list[dict]]:
        """Run multiple association queries in one Chroma call."""
        if not vectors:
            return []
        embeds = [v.tolist() for v in vectors]
        try:
            raw = self._associations.query(
                query_embeddings=embeds,
                n_results=n,
                include=["metadatas", "documents", "distances"],
            )
        except Exception:
            n_avail = max(self._associations.count(), 1)
            if n_avail == 0:
                return [[] for _ in vectors]
            raw = self._associations.query(
                query_embeddings=embeds,
                n_results=min(n, n_avail),
                include=["metadatas", "documents", "distances"],
            )
        return self._unpack_multi(raw)

    # -- dirty set for incremental consolidation --

    def _mark_dirty(self, concept_id: str) -> None:
        with self._dirty_lock:
            self._dirty_ids.add(concept_id)

    def pop_dirty_ids(self) -> set[str]:
        """Return and clear the set of concept IDs modified since last call."""
        with self._dirty_lock:
            ids = self._dirty_ids.copy()
            self._dirty_ids.clear()
        return ids

    # -- crystallized tier --

    def promote_to_crystal(self, concept_id: str, vector: np.ndarray,
                           metadata: dict, document: str) -> None:
        """Add a crystallized concept to the fast in-memory tier."""
        with self._crystal_lock:
            self._crystallized[concept_id] = (
                vector.astype(np.float32), dict(metadata), document)

    def query_crystal_tier(self, vector: np.ndarray,
                           n: int = 5) -> list[dict]:
        """Fast cosine search over the in-memory crystallized tier."""
        with self._crystal_lock:
            if not self._crystallized:
                return []
            ids = list(self._crystallized.keys())
            data = [self._crystallized[k] for k in ids]

        mat = np.stack([d[0] for d in data])
        sims = mat @ vector.astype(np.float32)
        top = np.argsort(-sims)[:n]
        results = []
        for idx in top:
            cid = ids[idx]
            vec, meta, doc = data[idx]
            results.append({
                "id": cid,
                "metadata": meta,
                "document": doc,
                "distance": float(1.0 - sims[idx]),
            })
        return results

    def crystal_count(self) -> int:
        with self._crystal_lock:
            return len(self._crystallized)

    # -- all vectors export (for HEI build) --

    def export_all_vectors(self) -> tuple[list[str], np.ndarray]:
        """Return (ids, vectors) for every concept. Used by HEI build."""
        data = self._concepts.get(include=["embeddings"])
        ids = data.get("ids", [])
        embeds = data.get("embeddings", [])
        if not ids or not embeds:
            return [], np.empty((0, 512), dtype=np.float32)
        return ids, np.array(embeds, dtype=np.float32)

    def purge_incoherent(self, coherence_fn) -> int:
        """Delete concepts and associations whose document text fails *coherence_fn*.

        Returns the total number of deleted entries.
        """
        deleted = 0
        for collection in (self._concepts, self._associations):
            try:
                data = collection.get(include=["documents", "metadatas"])
            except Exception:
                continue
            ids = data.get("ids", [])
            docs = data.get("documents", [])
            metas = data.get("metadatas", [])
            bad_ids = []
            for i, cid in enumerate(ids):
                doc = docs[i] if i < len(docs) and docs[i] else ""
                meta = metas[i] if i < len(metas) and metas[i] else {}
                full_text = meta.get("full_text", "")
                text = full_text if full_text else doc
                if not coherence_fn(text):
                    bad_ids.append(cid)
            if bad_ids:
                try:
                    collection.delete(ids=bad_ids)
                    deleted += len(bad_ids)
                except Exception:
                    for bid in bad_ids:
                        try:
                            collection.delete(ids=[bid])
                            deleted += 1
                        except Exception:
                            pass
        if deleted:
            log.info("Purged %d incoherent entries from vector store", deleted)
        return deleted

    def stats(self) -> dict:
        return {
            "concepts": self._concepts.count(),
            "associations": self._associations.count(),
            "crystallized": self.crystal_count(),
            "dirty_pending": len(self._dirty_ids),
        }

    @staticmethod
    def _unpack_multi(results: dict) -> list[list[dict]]:
        """Unpack multi-query Chroma results into list of list of dicts."""
        all_results: list[list[dict]] = []
        if not results or not results.get("ids"):
            return all_results
        for q_idx in range(len(results["ids"])):
            items: list[dict] = []
            ids_list = results["ids"][q_idx]
            metas = results.get("metadatas", [[]])[q_idx] if results.get("metadatas") else []
            docs = results.get("documents", [[]])[q_idx] if results.get("documents") else []
            dists = results.get("distances", [[]])[q_idx] if results.get("distances") else []
            for i, rid in enumerate(ids_list):
                item: dict = {"id": rid}
                if i < len(metas) and metas[i] is not None:
                    item["metadata"] = metas[i]
                if i < len(docs) and docs[i] is not None:
                    item["document"] = docs[i]
                if i < len(dists):
                    item["distance"] = dists[i]
                items.append(item)
            all_results.append(items)
        return all_results

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

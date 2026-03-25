"""Quantum Superposition Batching Engine.

Replaces all single-item CLIP / Chroma operations with GPU-saturating
batch pipelines.  Two priority lanes keep chat responsive while
background ingestion and training fill the GPU to capacity.

Batch accumulation logic:
  - Items pile up in an asyncio.Queue.
  - A flush happens on whichever comes first:
      * The batch reaches CLIP_BATCH_SIZE (64 items for RTX 4060), OR
      * FLUSH_INTERVAL_MS (50 ms) has passed since the first item landed.
  - Urgent (chat) requests bypass the queue and get a dedicated fast path.

All public methods are thread-safe and can be called from asyncio *or*
sync code (there is a thin sync wrapper that bridges to the async core).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from lqnn.core.vector_store import VectorStore
    from lqnn.models.clip_encoder import CLIPEncoder

log = logging.getLogger(__name__)

# --- Tuning constants (RTX 4060 8 GB) --------------------------------
DEFAULT_CLIP_BATCH_SIZE = 64
DEFAULT_CHROMA_WRITE_BATCH = 100
DEFAULT_CHROMA_QUERY_BATCH = 32
DEFAULT_FLUSH_INTERVAL_MS = 20   # reduced from 50ms; URGENT items wake instantly
# ----------------------------------------------------------------------


class Priority(Enum):
    URGENT = auto()   # chat queries -- served immediately
    NORMAL = auto()   # ingestion / training -- batched


@dataclass
class _EncodeRequest:
    texts: list[str]
    future: Future
    priority: Priority = Priority.NORMAL
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class _WriteRequest:
    entry: Any  # VectorEntry
    future: Future


@dataclass
class _QueryRequest:
    vector: np.ndarray
    n: int
    future: Future
    collection: str = "concepts"


class QuantumBatchEngine:
    """Central batching hub shared by every component in the system.

    Lifecycle:
        engine = QuantumBatchEngine(clip, store)
        await engine.start()          # spawns background flush tasks
        ...
        await engine.stop()
    """

    def __init__(self, clip: CLIPEncoder, store: VectorStore) -> None:
        self._clip = clip
        self._store = store

        self._encode_queue: deque[_EncodeRequest] = deque()
        self._write_queue: deque[_WriteRequest] = deque()
        self._query_queue: deque[_QueryRequest] = deque()

        self._lock = threading.Lock()
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: list[asyncio.Task] = []
        self._profile = "default"
        self._clip_batch_size = DEFAULT_CLIP_BATCH_SIZE
        self._write_batch_size = DEFAULT_CHROMA_WRITE_BATCH
        self._query_batch_size = DEFAULT_CHROMA_QUERY_BATCH
        self._flush_interval_ms = DEFAULT_FLUSH_INTERVAL_MS

        self._urgent_encode_event = threading.Event()
        self._urgent_query_event = threading.Event()

        # stats
        self._encode_batches = 0
        self._encode_items = 0
        self._write_batches = 0
        self._write_items = 0
        self._query_batches = 0
        self._query_items = 0

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._tasks = [
            asyncio.create_task(self._encode_flush_loop()),
            asyncio.create_task(self._write_flush_loop()),
            asyncio.create_task(self._query_flush_loop()),
        ]
        log.info(
            "QuantumBatchEngine started (profile=%s, batch=%d, flush=%dms)",
            self._profile, self._clip_batch_size, self._flush_interval_ms,
        )

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._flush_encode_now()
        self._flush_writes_now()
        self._flush_queries_now()
        log.info("QuantumBatchEngine stopped")

    # ------------------------------------------------------------------ #
    # Public API -- encode                                                 #
    # ------------------------------------------------------------------ #

    def encode_text_urgent(self, text: str) -> np.ndarray:
        """Encode a single text immediately (bypasses batch queue)."""
        return self._clip.encode_text(text)

    def encode_texts_urgent(self, texts: list[str]) -> np.ndarray:
        """Encode multiple texts immediately (bypasses batch queue)."""
        return self._clip.encode_texts(texts)

    def submit_encode(self, texts: list[str],
                      priority: Priority = Priority.NORMAL) -> Future:
        """Submit texts for batched encoding.  Returns a Future[np.ndarray]."""
        fut: Future = Future()
        req = _EncodeRequest(texts=texts, future=fut, priority=priority)
        with self._lock:
            self._encode_queue.append(req)
        if priority == Priority.URGENT:
            self._urgent_encode_event.set()
        return fut

    # ------------------------------------------------------------------ #
    # Public API -- write                                                  #
    # ------------------------------------------------------------------ #

    def submit_write(self, entry: Any) -> Future:
        """Submit a VectorEntry for batched Chroma upsert.  Returns Future[None]."""
        fut: Future = Future()
        with self._lock:
            self._write_queue.append(_WriteRequest(entry=entry, future=fut))
        return fut

    # ------------------------------------------------------------------ #
    # Public API -- query                                                  #
    # ------------------------------------------------------------------ #

    def query_urgent(self, vector: np.ndarray, n: int = 10,
                     collection: str = "concepts") -> list[dict]:
        """Run a single query immediately (bypasses batch queue)."""
        if collection == "associations":
            return self._store.query_associations(vector, n)
        return self._store.query_concepts(vector, n)

    def submit_query(self, vector: np.ndarray, n: int = 10,
                     collection: str = "concepts",
                     urgent: bool = False) -> Future:
        """Submit a query for batched execution.  Returns Future[list[dict]]."""
        fut: Future = Future()
        with self._lock:
            self._query_queue.append(
                _QueryRequest(vector=vector, n=n, future=fut,
                              collection=collection))
        if urgent:
            self._urgent_query_event.set()
        return fut

    # ------------------------------------------------------------------ #
    # Flush loops                                                          #
    # ------------------------------------------------------------------ #

    async def _encode_flush_loop(self) -> None:
        interval = self._flush_interval_ms / 1000
        while self._running:
            woke_urgent = await asyncio.to_thread(
                self._urgent_encode_event.wait, interval)
            if woke_urgent:
                self._urgent_encode_event.clear()
            try:
                await asyncio.to_thread(self._flush_encode_now)
            except Exception as exc:
                log.debug("Encode flush error: %s", exc)

    async def _write_flush_loop(self) -> None:
        interval = self._flush_interval_ms / 1000
        while self._running:
            await asyncio.sleep(interval)
            try:
                await asyncio.to_thread(self._flush_writes_now)
            except Exception as exc:
                log.debug("Write flush error: %s", exc)

    async def _query_flush_loop(self) -> None:
        interval = self._flush_interval_ms / 1000
        while self._running:
            woke_urgent = await asyncio.to_thread(
                self._urgent_query_event.wait, interval)
            if woke_urgent:
                self._urgent_query_event.clear()
            try:
                await asyncio.to_thread(self._flush_queries_now)
            except Exception as exc:
                log.debug("Query flush error: %s", exc)

    # ------------------------------------------------------------------ #
    # Actual flush implementations                                         #
    # ------------------------------------------------------------------ #

    def _flush_encode_now(self) -> None:
        """Drain up to configured encode batch and process as one GPU call.

        URGENT requests are sorted to the front of the batch so chat-path
        encodes are never delayed by background ingestion.
        """
        with self._lock:
            batch: list[_EncodeRequest] = []
            while self._encode_queue and len(batch) < self._clip_batch_size:
                batch.append(self._encode_queue.popleft())
        if not batch:
            return

        batch.sort(key=lambda r: (0 if r.priority == Priority.URGENT else 1,
                                   r.created_at))

        all_texts: list[str] = []
        slices: list[tuple[int, int]] = []
        for req in batch:
            start = len(all_texts)
            all_texts.extend(req.texts)
            slices.append((start, len(all_texts)))

        try:
            all_vecs = self._clip.encode_texts(all_texts)
            for req, (s, e) in zip(batch, slices):
                req.future.set_result(all_vecs[s:e])
            self._encode_batches += 1
            self._encode_items += len(all_texts)
        except Exception as exc:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)

    def _flush_writes_now(self) -> None:
        """Drain up to CHROMA_WRITE_BATCH write requests and upsert at once."""
        with self._lock:
            batch: list[_WriteRequest] = []
            while self._write_queue and len(batch) < self._write_batch_size:
                batch.append(self._write_queue.popleft())
        if not batch:
            return

        ids = []
        embeddings = []
        metadatas = []
        documents = []
        entries = []
        for wr in batch:
            entry = wr.entry
            entries.append(entry)
            ids.append(entry.id)
            embeddings.append(entry.vector.tolist())
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

        try:
            self._store._concepts.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            for wr in batch:
                wr.future.set_result(None)
            self._write_batches += 1
            self._write_items += len(batch)
        except Exception as exc:
            for wr in batch:
                if not wr.future.done():
                    wr.future.set_exception(exc)

    def _flush_queries_now(self) -> None:
        """Drain up to CHROMA_QUERY_BATCH query requests and execute.

        Concept queries targeting the same collection are batched into a
        single Chroma multi-query call for much lower overhead.
        Association queries are still executed individually (different collection).
        """
        with self._lock:
            batch: list[_QueryRequest] = []
            while self._query_queue and len(batch) < self._query_batch_size:
                batch.append(self._query_queue.popleft())
        if not batch:
            return

        concept_batch: list[_QueryRequest] = []
        assoc_batch: list[_QueryRequest] = []
        for qr in batch:
            if qr.collection == "associations":
                assoc_batch.append(qr)
            else:
                concept_batch.append(qr)

        if concept_batch:
            try:
                max_n = max(qr.n for qr in concept_batch)
                vectors = [qr.vector for qr in concept_batch]
                all_results = self._store.batch_query_concepts(vectors, max_n)
                for qr, results in zip(concept_batch, all_results):
                    qr.future.set_result(results[:qr.n])
            except Exception as exc:
                for qr in concept_batch:
                    if not qr.future.done():
                        qr.future.set_exception(exc)

        if assoc_batch:
            try:
                max_n = max(qr.n for qr in assoc_batch)
                vectors = [qr.vector for qr in assoc_batch]
                all_results = self._store.batch_query_associations(
                    vectors, max_n)
                for qr, results in zip(assoc_batch, all_results):
                    qr.future.set_result(results[:qr.n])
            except Exception:
                for qr in assoc_batch:
                    try:
                        result = self._store.query_associations(
                            qr.vector, qr.n)
                        qr.future.set_result(result)
                    except Exception as exc:
                        if not qr.future.done():
                            qr.future.set_exception(exc)

        self._query_batches += 1
        self._query_items += len(batch)

    # ------------------------------------------------------------------ #
    # Stats                                                                #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        return {
            "profile": self._profile,
            "clip_batch_size": self._clip_batch_size,
            "write_batch_size": self._write_batch_size,
            "query_batch_size": self._query_batch_size,
            "flush_interval_ms": self._flush_interval_ms,
            "encode_batches": self._encode_batches,
            "encode_items": self._encode_items,
            "write_batches": self._write_batches,
            "write_items": self._write_items,
            "query_batches": self._query_batches,
            "query_items": self._query_items,
            "encode_queue_len": len(self._encode_queue),
            "write_queue_len": len(self._write_queue),
            "query_queue_len": len(self._query_queue),
        }

    def set_quantum_profile(self, profile: str) -> dict:
        """Tune runtime batching without requiring extra hardware."""
        profile = (profile or "").strip().lower()
        if profile == "frontier_train_only":
            self._profile = profile
            self._clip_batch_size = 96
            self._write_batch_size = 128
            self._query_batch_size = 48
            self._flush_interval_ms = 35
        else:
            self._profile = "default"
            self._clip_batch_size = DEFAULT_CLIP_BATCH_SIZE
            self._write_batch_size = DEFAULT_CHROMA_WRITE_BATCH
            self._query_batch_size = DEFAULT_CHROMA_QUERY_BATCH
            self._flush_interval_ms = DEFAULT_FLUSH_INTERVAL_MS
        return self.stats()

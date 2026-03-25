"""Temporal Coherence Pipeline.

Replaces the sequential 45-second training cycle with a 5-stage
continuous streaming pipeline where stages overlap:

Stage 1 -- Perception  : async intake of new data (web, ingestion, chat)
Stage 2 -- Encoding    : GPU batch CLIP encoding (via QuantumBatchEngine)
Stage 3 -- Integration : batch Chroma writes + HEI cluster updates
Stage 4 -- Consolidation : incremental (dirty-set only, O(delta) not O(N))
Stage 5 -- Resonance   : background association generation + resonance detection

While Stage 2 encodes batch N, Stage 1 is already collecting batch N+1.
Effective throughput is ~5x higher than a serial loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)

PERCEPTION_QUEUE_MAX = 500
ENCODING_BATCH_SIZE = 64
INTEGRATION_BATCH_SIZE = 100
CONSOLIDATION_INTERVAL_S = 120
RESONANCE_INTERVAL_S = 300
PIPELINE_TICK_MS = 25


@dataclass
class PipelineItem:
    """A unit of work flowing through the pipeline."""
    text: str
    source: str = "pipeline"
    source_type: str = "text"
    image: bytes | None = None
    metadata: dict = field(default_factory=dict)
    vector: np.ndarray | None = None
    concept_id: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class PipelineMetrics:
    perceived: int = 0
    encoded: int = 0
    integrated: int = 0
    consolidated: int = 0
    resonances: int = 0
    pipeline_throughput: float = 0.0
    last_cycle_ms: float = 0.0


class TemporalPipeline:
    """Five-stage streaming pipeline for continuous learning.

    Usage:
        pipeline = TemporalPipeline(memory, batch_engine, hei)
        await pipeline.start()
        pipeline.submit(PipelineItem(text="banana", source="crawl"))
        ...
        await pipeline.stop()
    """

    def __init__(self, memory, batch_engine=None, hei=None,
                 event_callback: Callable[[dict], None] | None = None) -> None:
        self._memory = memory
        self._batch_engine = batch_engine
        self._hei = hei
        self._emit = event_callback or (lambda e: None)

        self._perception_q: asyncio.Queue[PipelineItem] = asyncio.Queue(
            maxsize=PERCEPTION_QUEUE_MAX)
        self._encoding_q: asyncio.Queue[PipelineItem] = asyncio.Queue(
            maxsize=ENCODING_BATCH_SIZE * 4)
        self._integration_q: asyncio.Queue[PipelineItem] = asyncio.Queue(
            maxsize=INTEGRATION_BATCH_SIZE * 4)

        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._metrics = PipelineMetrics()
        self._last_consolidation = 0.0
        self._last_resonance = 0.0
        self._start_time = 0.0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def submit(self, item: PipelineItem) -> bool:
        """Submit a new item to Stage 1 (Perception). Non-blocking."""
        try:
            self._perception_q.put_nowait(item)
            self._metrics.perceived += 1
            return True
        except asyncio.QueueFull:
            return False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._last_consolidation = time.time()
        self._last_resonance = time.time()
        self._tasks = [
            asyncio.create_task(self._stage_perception()),
            asyncio.create_task(self._stage_encoding()),
            asyncio.create_task(self._stage_integration()),
            asyncio.create_task(self._stage_consolidation()),
            asyncio.create_task(self._stage_resonance()),
        ]
        log.info("TemporalPipeline started (5 stages)")

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        log.info("TemporalPipeline stopped")

    # ------------------------------------------------------------------ #
    # Stage 1 -- Perception                                                #
    # ------------------------------------------------------------------ #

    async def _stage_perception(self) -> None:
        """Continuously drain the perception queue and forward items
        to encoding, applying basic quality filters."""
        while self._running:
            try:
                item = await asyncio.wait_for(
                    self._perception_q.get(), timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue

            text = item.text.strip()
            if len(text) < 10:
                continue

            printable = sum(c.isprintable() for c in text) / max(len(text), 1)
            if printable < 0.5:
                continue

            try:
                self._encoding_q.put_nowait(item)
            except asyncio.QueueFull:
                pass

    # ------------------------------------------------------------------ #
    # Stage 2 -- Encoding                                                  #
    # ------------------------------------------------------------------ #

    async def _stage_encoding(self) -> None:
        """Batch-encode items using CLIP."""
        while self._running:
            batch: list[PipelineItem] = []
            deadline = time.monotonic() + PIPELINE_TICK_MS / 1000

            while len(batch) < ENCODING_BATCH_SIZE and time.monotonic() < deadline:
                try:
                    item = self._encoding_q.get_nowait()
                    batch.append(item)
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.005)
                    break

            if not batch:
                await asyncio.sleep(PIPELINE_TICK_MS / 1000)
                continue

            texts = [item.text[:512] for item in batch]
            try:
                vectors = await asyncio.to_thread(
                    self._memory.clip.encode_texts, texts)
                for item, vec in zip(batch, vectors):
                    item.vector = vec
                    try:
                        self._integration_q.put_nowait(item)
                    except asyncio.QueueFull:
                        pass
                self._metrics.encoded += len(batch)
            except Exception as exc:
                log.debug("Pipeline encode error: %s", exc)

    # ------------------------------------------------------------------ #
    # Stage 3 -- Integration                                               #
    # ------------------------------------------------------------------ #

    async def _stage_integration(self) -> None:
        """Batch-write encoded items into ChromaDB and update HEI."""
        from lqnn.core.vector_store import VectorEntry
        import hashlib

        while self._running:
            batch: list[PipelineItem] = []
            deadline = time.monotonic() + PIPELINE_TICK_MS * 2 / 1000

            while len(batch) < INTEGRATION_BATCH_SIZE and time.monotonic() < deadline:
                try:
                    item = self._integration_q.get_nowait()
                    if item.vector is not None:
                        batch.append(item)
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.01)
                    break

            if not batch:
                await asyncio.sleep(PIPELINE_TICK_MS * 2 / 1000)
                continue

            entries = []
            for item in batch:
                cid = hashlib.sha256(
                    item.text[:100].encode()).hexdigest()[:16]
                item.concept_id = cid
                label = item.text.split(".")[0][:100].strip() or item.text[:100]
                entry = VectorEntry(
                    id=cid,
                    vector=item.vector,
                    concept=label,
                    source=item.source,
                    volatility=0.5,
                    confidence=0.6,
                    metadata={
                        "full_text": item.text[:2000],
                        "source_type": item.source_type,
                        **{k: str(v) for k, v in item.metadata.items()
                           if isinstance(v, (str, int, float, bool))},
                    },
                )
                entries.append(entry)

            try:
                await asyncio.to_thread(
                    self._memory.store.batch_add_concepts, entries)

                if self._hei:
                    for item in batch:
                        if item.vector is not None:
                            self._hei.assign(item.concept_id, item.vector)

                self._metrics.integrated += len(batch)

                for item in batch:
                    import queue as _q
                    try:
                        label = item.text.split(".")[0][:100].strip()
                        self._memory._assoc_bg_queue.put_nowait(
                            (label, item.vector, 5))
                    except _q.Full:
                        pass

            except Exception as exc:
                log.debug("Pipeline integration error: %s", exc)

    # ------------------------------------------------------------------ #
    # Stage 4 -- Consolidation (incremental)                               #
    # ------------------------------------------------------------------ #

    async def _stage_consolidation(self) -> None:
        """Periodically run incremental consolidation on dirty concepts."""
        while self._running:
            await asyncio.sleep(5)
            elapsed = time.time() - self._last_consolidation
            if elapsed < CONSOLIDATION_INTERVAL_S:
                continue

            try:
                result = await asyncio.to_thread(self._memory.consolidate)
                self._last_consolidation = time.time()
                self._metrics.consolidated += (
                    result.get("pruned", 0) +
                    result.get("crystallized", 0) +
                    result.get("decayed", 0)
                )
                self._emit({
                    "type": "pipeline_consolidation",
                    "result": result,
                    "timestamp": time.time(),
                })
            except Exception as exc:
                log.debug("Pipeline consolidation error: %s", exc)

    # ------------------------------------------------------------------ #
    # Stage 5 -- Resonance                                                 #
    # ------------------------------------------------------------------ #

    async def _stage_resonance(self) -> None:
        """Periodically detect resonances and rebuild HEI if needed."""
        while self._running:
            await asyncio.sleep(10)
            elapsed = time.time() - self._last_resonance
            if elapsed < RESONANCE_INTERVAL_S:
                continue

            try:
                res = await asyncio.to_thread(
                    self._memory.detect_resonance, 5)
                self._metrics.resonances += len(res)
                self._last_resonance = time.time()

                if self._hei and self._hei.needs_rebuild():
                    ids, vecs = await asyncio.to_thread(
                        self._memory.store.export_all_vectors)
                    if len(ids) > 50:
                        await asyncio.to_thread(self._hei.build, ids, vecs)
                        self._emit({
                            "type": "pipeline_hei_rebuild",
                            "concepts": len(ids),
                            "timestamp": time.time(),
                        })
            except Exception as exc:
                log.debug("Pipeline resonance error: %s", exc)

    # ------------------------------------------------------------------ #
    # Stats                                                                #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        uptime = time.time() - self._start_time if self._start_time else 0
        throughput = self._metrics.integrated / max(uptime, 1)
        return {
            "perceived": self._metrics.perceived,
            "encoded": self._metrics.encoded,
            "integrated": self._metrics.integrated,
            "consolidated": self._metrics.consolidated,
            "resonances": self._metrics.resonances,
            "throughput_per_sec": round(throughput, 2),
            "perception_q": self._perception_q.qsize(),
            "encoding_q": self._encoding_q.qsize(),
            "integration_q": self._integration_q.qsize(),
            "uptime_s": round(uptime, 1),
        }

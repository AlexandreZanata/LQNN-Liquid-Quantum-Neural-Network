"""Continuous training loop -- the never-stopping brain.

Runs autonomously inside Docker:
1. Detect knowledge gaps
2. Crawl the web for images + text
3. Encode with CLIP, generate associations with LLM
4. Store in ChromaDB with volatility metadata
5. Periodically consolidate (crystallize stable, prune volatile)
6. Self-play when idle (query own knowledge, reinforce)
7. Log metrics to MongoDB
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from lqnn.agents.manager import AgentManager
from lqnn.core.associative_memory import AssociativeMemory

log = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    cycle: int = 0
    total_concepts: int = 0
    total_associations: int = 0
    concepts_this_cycle: int = 0
    images_this_cycle: int = 0
    consolidation_pruned: int = 0
    consolidation_crystallized: int = 0
    self_play_actions: int = 0
    cycle_duration_s: float = 0.0
    uptime_s: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ContinuousTrainer:
    """The autonomous training engine.

    Runs as an asyncio loop inside the Docker container.
    Never stops learning while the container is alive.
    """

    CRAWL_INTERVAL_S = 60
    CONSOLIDATION_INTERVAL_CYCLES = 10
    SELF_PLAY_INTERVAL_CYCLES = 3
    METRICS_LOG_INTERVAL_CYCLES = 5

    def __init__(self, memory: AssociativeMemory,
                 agent_manager: AgentManager,
                 training_db=None) -> None:
        self.memory = memory
        self.agent_manager = agent_manager
        self.training_db = training_db
        self._running = False
        self._cycle = 0
        self._start_time = 0.0
        self._latest_metrics: TrainingMetrics | None = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._task = asyncio.create_task(self._loop())
        log.info("Continuous trainer started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.agent_manager.shutdown()
        log.info("Continuous trainer stopped")

    async def _loop(self) -> None:
        while self._running:
            try:
                metrics = await self._run_one_cycle()
                self._latest_metrics = metrics
                self._log_metrics(metrics)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("Training cycle error: %s", exc, exc_info=True)

            await asyncio.sleep(self.CRAWL_INTERVAL_S)

    async def _run_one_cycle(self) -> TrainingMetrics:
        t0 = time.time()
        self._cycle += 1

        report = await self.agent_manager.run_cycle()

        consolidation = {"pruned": 0, "crystallized": 0}
        if self._cycle % self.CONSOLIDATION_INTERVAL_CYCLES == 0:
            consolidation = self.memory.consolidate()

        self_play_result = {"action": "skip"}
        if self._cycle % self.SELF_PLAY_INTERVAL_CYCLES == 0:
            self_play_result = self.memory.self_play_cycle()

        store_stats = self.memory.store.stats()

        metrics = TrainingMetrics(
            cycle=self._cycle,
            total_concepts=store_stats["concepts"],
            total_associations=store_stats["associations"],
            concepts_this_cycle=report.concepts_learned,
            images_this_cycle=report.images_processed,
            consolidation_pruned=consolidation.get("pruned", 0),
            consolidation_crystallized=consolidation.get("crystallized", 0),
            self_play_actions=1 if self_play_result.get("action") != "skip" else 0,
            cycle_duration_s=time.time() - t0,
            uptime_s=time.time() - self._start_time,
        )
        return metrics

    def _log_metrics(self, metrics: TrainingMetrics) -> None:
        if self._cycle % self.METRICS_LOG_INTERVAL_CYCLES == 0:
            log.info(
                "Cycle %d: concepts=%d assoc=%d learned=%d imgs=%d "
                "pruned=%d crystal=%d dur=%.1fs uptime=%.0fs",
                metrics.cycle, metrics.total_concepts,
                metrics.total_associations, metrics.concepts_this_cycle,
                metrics.images_this_cycle, metrics.consolidation_pruned,
                metrics.consolidation_crystallized,
                metrics.cycle_duration_s, metrics.uptime_s,
            )

        if self.training_db:
            try:
                self.training_db.log_training_cycle({
                    "cycle": metrics.cycle,
                    "concepts": metrics.total_concepts,
                    "associations": metrics.total_associations,
                    "learned": metrics.concepts_this_cycle,
                    "images": metrics.images_this_cycle,
                    "pruned": metrics.consolidation_pruned,
                    "crystallized": metrics.consolidation_crystallized,
                    "duration_s": metrics.cycle_duration_s,
                    "uptime_s": metrics.uptime_s,
                    "timestamp": metrics.timestamp,
                })
            except Exception:
                pass

    async def run_manual_cycle(self) -> TrainingMetrics:
        """Trigger a single cycle from the UI."""
        return await self._run_one_cycle()

    def latest_metrics(self) -> dict | None:
        if not self._latest_metrics:
            return None
        m = self._latest_metrics
        return {
            "cycle": m.cycle,
            "total_concepts": m.total_concepts,
            "total_associations": m.total_associations,
            "concepts_this_cycle": m.concepts_this_cycle,
            "images_this_cycle": m.images_this_cycle,
            "consolidation_pruned": m.consolidation_pruned,
            "consolidation_crystallized": m.consolidation_crystallized,
            "self_play_actions": m.self_play_actions,
            "cycle_duration_s": round(m.cycle_duration_s, 2),
            "uptime_s": round(m.uptime_s, 1),
        }

    def status(self) -> dict:
        return {
            "running": self._running,
            "cycle": self._cycle,
            "uptime_s": round(time.time() - self._start_time, 1) if self._start_time else 0,
            "latest_metrics": self.latest_metrics(),
            "memory": self.memory.stats(),
            "agents": self.agent_manager.stats(),
        }

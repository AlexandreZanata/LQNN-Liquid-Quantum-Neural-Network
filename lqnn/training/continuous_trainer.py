"""Continuous training loop -- the never-stopping quantum brain.

Runs autonomously inside Docker:
Phase 1 (cycles 1-100): Visual Objects -- learn concrete objects with images
Phase 2 (cycles 100+): Abstract Concepts -- expand to science, history, etc.
Phase Continuous: Self-evolution -- IA decides what to learn next

v2: Integrates with TemporalPipeline for overlapping stage execution.
Consolidation and resonance are delegated to the pipeline when available,
while the trainer still owns phased web crawling and self-play.

Each cycle:
1. Detect knowledge gaps (phase-aware)
2. Crawl the web for images + text
3. Encode with CLIP, generate associations with LLM
4. Store in ChromaDB with volatility metadata
5. Periodically consolidate (crystallize stable, prune volatile)
6. Self-play when idle (query own knowledge, reinforce)
7. Log metrics to MongoDB
8. Broadcast events to UI via callback
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from lqnn.agents.manager import AgentManager
from lqnn.core.associative_memory import AssociativeMemory

log = logging.getLogger(__name__)


class TrainingPhase(Enum):
    VISUAL_OBJECTS = "visual_objects"
    ABSTRACT_CONCEPTS = "abstract_concepts"
    SELF_EVOLUTION = "self_evolution"


VISUAL_SEEDS = [
    "banana", "apple", "cat", "dog", "car", "tree", "sun", "moon",
    "house", "flower", "fish", "bird", "mountain", "river", "ocean",
    "fire", "water", "rock", "cloud", "rain", "snow", "butterfly",
    "horse", "elephant", "lion", "guitar", "piano", "book", "pencil",
    "chair", "table", "computer", "phone", "clock", "key", "door",
    "window", "bridge", "train", "airplane", "boat", "bicycle",
    "shoe", "hat", "bread", "egg", "milk", "coffee", "rose", "diamond",
]

ABSTRACT_SEEDS = [
    "gravity", "evolution", "democracy", "mathematics", "philosophy",
    "music", "language", "emotion", "memory", "consciousness",
    "energy", "time", "space", "light", "sound", "color",
    "love", "fear", "happiness", "intelligence", "creativity",
    "history", "culture", "science", "technology", "medicine",
    "economics", "psychology", "physics", "chemistry", "biology",
    "astronomy", "geography", "literature", "art", "architecture",
]

# 20 benchmark questions to measure real AI progress.
# These cover math, algorithms, and CS fundamentals that have
# well-known, verifiable correct answers.
BENCHMARK_QUESTIONS = [
    {
        "id": "bm01",
        "question": "What is the time complexity of binary search?",
        "answer": "O(log n)",
        "category": "algorithms",
    },
    {
        "id": "bm02",
        "question": "What is the derivative of x^n with respect to x?",
        "answer": "n * x^(n-1)",
        "category": "calculus",
    },
    {
        "id": "bm03",
        "question": "What is the time complexity of quicksort in the average case?",
        "answer": "O(n log n)",
        "category": "algorithms",
    },
    {
        "id": "bm04",
        "question": "What is the Pythagorean theorem formula?",
        "answer": "a^2 + b^2 = c^2",
        "category": "geometry",
    },
    {
        "id": "bm05",
        "question": "What is the integral of 1/x dx?",
        "answer": "ln|x| + C",
        "category": "calculus",
    },
    {
        "id": "bm06",
        "question": "What data structure uses LIFO (Last In First Out)?",
        "answer": "Stack",
        "category": "data_structures",
    },
    {
        "id": "bm07",
        "question": "What is Euler's identity equation?",
        "answer": "e^(i*pi) + 1 = 0",
        "category": "mathematics",
    },
    {
        "id": "bm08",
        "question": "What is the space complexity of merge sort?",
        "answer": "O(n)",
        "category": "algorithms",
    },
    {
        "id": "bm09",
        "question": "What is the quadratic formula for solving ax^2 + bx + c = 0?",
        "answer": "x = (-b +/- sqrt(b^2 - 4ac)) / (2a)",
        "category": "algebra",
    },
    {
        "id": "bm10",
        "question": "What algorithm finds the shortest path in a weighted graph?",
        "answer": "Dijkstra's algorithm",
        "category": "graph_algorithms",
    },
    {
        "id": "bm11",
        "question": "What is the Big-O complexity of accessing an element in a hash table?",
        "answer": "O(1) average case",
        "category": "data_structures",
    },
    {
        "id": "bm12",
        "question": "What is the fundamental theorem of calculus?",
        "answer": "The integral of f(x) from a to b equals F(b) - F(a) where F is the antiderivative of f",
        "category": "calculus",
    },
    {
        "id": "bm13",
        "question": "What is the time complexity of BFS and DFS graph traversal?",
        "answer": "O(V + E) where V is vertices and E is edges",
        "category": "graph_algorithms",
    },
    {
        "id": "bm14",
        "question": "What is the P vs NP problem in computer science?",
        "answer": "Whether every problem whose solution can be verified in polynomial time can also be solved in polynomial time",
        "category": "computational_complexity",
    },
    {
        "id": "bm15",
        "question": "What is dynamic programming and when should it be used?",
        "answer": "A method for solving complex problems by breaking them into overlapping subproblems with optimal substructure, using memoization or tabulation",
        "category": "algorithms",
    },
    {
        "id": "bm16",
        "question": "What is the Fibonacci sequence formula using recursion?",
        "answer": "F(n) = F(n-1) + F(n-2) with F(0)=0 and F(1)=1",
        "category": "mathematics",
    },
    {
        "id": "bm17",
        "question": "What is a balanced binary search tree and give an example?",
        "answer": "A BST where the height difference between left and right subtrees is at most 1 for every node. Examples: AVL tree, Red-Black tree",
        "category": "data_structures",
    },
    {
        "id": "bm18",
        "question": "What is the halting problem and why is it undecidable?",
        "answer": "The problem of determining whether a program will finish running or loop forever. It is undecidable because no algorithm can solve it for all possible program-input pairs, proven by Alan Turing via diagonalization",
        "category": "computability",
    },
    {
        "id": "bm19",
        "question": "What is the Master Theorem for solving recurrence relations?",
        "answer": "For T(n) = aT(n/b) + O(n^d): if d < log_b(a) then T(n) = O(n^log_b(a)); if d = log_b(a) then T(n) = O(n^d * log n); if d > log_b(a) then T(n) = O(n^d)",
        "category": "algorithms",
    },
    {
        "id": "bm20",
        "question": "What is the difference between a process and a thread in operating systems?",
        "answer": "A process is an independent execution unit with its own memory space. A thread is a lightweight unit within a process that shares the same memory space. Threads are faster to create and context-switch but share resources",
        "category": "operating_systems",
    },
]

BENCHMARK_SEEDS = [bm["question"] for bm in BENCHMARK_QUESTIONS]

PHASE_TRANSITION_CYCLE = 100
SELF_EVOLUTION_MIN_CONCEPTS = 200


@dataclass
class TrainingMetrics:
    cycle: int = 0
    phase: str = ""
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
    """The autonomous training engine with phased learning.

    Runs as an asyncio loop inside the Docker container.
    Never stops learning while the container is alive.
    """

    CRAWL_INTERVAL_S = 45
    CONSOLIDATION_INTERVAL_CYCLES = 10
    SELF_PLAY_INTERVAL_CYCLES = 3
    METRICS_LOG_INTERVAL_CYCLES = 5

    STATE_FILE = "data/state/trainer_state.json"

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
        self._event_callback = None
        self._training_log: list[dict] = []
        self._temporal_pipeline = None
        self._hei = None
        self._load_state()

    def set_temporal_pipeline(self, pipeline) -> None:
        """Inject the TemporalPipeline for overlapping stage execution."""
        self._temporal_pipeline = pipeline

    def set_hei(self, hei) -> None:
        """Inject HEI for periodic rebuilds."""
        self._hei = hei

    def set_event_callback(self, callback) -> None:
        """Set a callback for broadcasting training events to the UI."""
        self._event_callback = callback

    def _emit(self, event_type: str, data: dict) -> None:
        entry = {
            "type": event_type,
            "cycle": self._cycle,
            "timestamp": time.time(),
            **data,
        }
        self._training_log.append(entry)
        if len(self._training_log) > 500:
            self._training_log = self._training_log[-250:]
        if self._event_callback:
            try:
                self._event_callback(entry)
            except Exception:
                pass

    @property
    def training_log(self) -> list[dict]:
        return self._training_log

    def _save_state(self) -> None:
        """Persist cycle and phase to disk so they survive container restarts."""
        import json
        import os
        try:
            os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)
            state = {
                "cycle": self._cycle,
                "phase": self.current_phase().value,
                "agent_cycle": self.agent_manager._cycle,
                "timestamp": time.time(),
            }
            if self._latest_metrics:
                state["last_metrics"] = {
                    "total_concepts": self._latest_metrics.total_concepts,
                    "total_associations": self._latest_metrics.total_associations,
                }
            with open(self.STATE_FILE, "w") as f:
                json.dump(state, f)
        except Exception as e:
            log.debug("Failed to save trainer state: %s", e)

    def _load_state(self) -> None:
        """Restore cycle/phase from disk if available."""
        import json
        import os
        try:
            if os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, "r") as f:
                    state = json.load(f)
                self._cycle = state.get("cycle", 0)
                agent_cycle = state.get("agent_cycle", 0)
                if hasattr(self.agent_manager, '_cycle'):
                    self.agent_manager._cycle = agent_cycle
                log.info("Restored trainer state: cycle=%d, agent_cycle=%d",
                         self._cycle, agent_cycle)
        except Exception as e:
            log.debug("Failed to load trainer state: %s", e)

    def current_phase(self) -> TrainingPhase:
        concept_count = self.memory.store.concept_count()
        if self._cycle < PHASE_TRANSITION_CYCLE:
            return TrainingPhase.VISUAL_OBJECTS
        if concept_count < SELF_EVOLUTION_MIN_CONCEPTS:
            return TrainingPhase.ABSTRACT_CONCEPTS
        return TrainingPhase.SELF_EVOLUTION

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_time = time.time()

        if self._temporal_pipeline:
            await self._temporal_pipeline.start()
            self._emit("trainer_start", {
                "message": "Continuous training started (temporal pipeline active)",
            })
        else:
            self._emit("trainer_start", {"message": "Continuous training started"})

        self._task = asyncio.create_task(self._loop())
        log.info("Continuous trainer started")

    async def stop(self) -> None:
        self._running = False
        if self._temporal_pipeline:
            await self._temporal_pipeline.stop()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.agent_manager.shutdown()
        self._emit("trainer_stop", {"message": "Continuous training stopped"})
        log.info("Continuous trainer stopped")

    async def _warmup_models(self) -> None:
        """Load CLIP and LLM in a background thread so the event loop stays free."""
        self._emit("warmup", {
            "stage": "start",
            "message": "Loading AI models (may download on first run)...",
        })

        try:
            cache = await asyncio.to_thread(self.memory.llm.cache_status)
            if cache.get("cached"):
                self._emit("warmup", {
                    "stage": "cache_check",
                    "message": (
                        f"Qwen cache found: {cache.get('size_gb', 0)}GB, "
                        f"{cache.get('shards', 0)} shards"
                    ),
                })
            else:
                self._emit("warmup", {
                    "stage": "cache_check",
                    "message": "Qwen cache missing/partial: first load will download model files",
                })
        except Exception:
            self._emit("warmup", {
                "stage": "cache_check",
                "message": "Could not inspect Qwen cache; continuing with normal load",
            })

        try:
            t0 = time.time()
            self._emit("warmup", {
                "stage": "clip_load",
                "message": "Loading CLIP model...",
            })
            await asyncio.to_thread(self.memory.clip.load)
            self._emit("warmup", {
                "stage": "clip_ready",
                "message": f"CLIP model loaded ({time.time() - t0:.1f}s)",
            })
        except Exception as exc:
            self._emit("error", {"message": f"CLIP load failed: {exc}"})
            log.error("CLIP warmup failed: %s", exc)

        try:
            t0 = time.time()
            self._emit("warmup", {
                "stage": "llm_load",
                "message": "Loading Qwen LLM (download or cache load)...",
            })
            await asyncio.to_thread(self.memory.llm.load)
            self._emit("warmup", {
                "stage": "llm_ready",
                "message": f"LLM model loaded ({time.time() - t0:.1f}s)",
            })
        except Exception as exc:
            self._emit("error", {"message": f"LLM load failed: {exc}"})
            log.error("LLM warmup failed: %s", exc)

        self._emit("warmup", {
            "stage": "inference_ready",
            "message": "Inference ready: model loaded on device and available for chat/training",
        })

    async def _loop(self) -> None:
        await self._warmup_models()

        while self._running:
            try:
                metrics = await self._run_one_cycle()
                self._latest_metrics = metrics
                self._log_metrics(metrics)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("Training cycle error: %s", exc, exc_info=True)
                self._emit("error", {"message": str(exc)})

            await asyncio.sleep(self.CRAWL_INTERVAL_S)

    async def _run_one_cycle(self) -> TrainingMetrics:
        t0 = time.time()
        self._cycle += 1
        phase = self.current_phase()

        self._emit("cycle_start", {
            "phase": phase.value,
            "message": f"Cycle {self._cycle} starting ({phase.value})",
        })

        self.agent_manager.set_phase(phase)
        report = await self.agent_manager.run_cycle()

        self._emit("cycle_crawl", {
            "concepts_learned": report.concepts_learned,
            "images_processed": report.images_processed,
            "gaps_resolved": report.gaps_resolved,
        })

        consolidation = {"pruned": 0, "crystallized": 0}
        use_pipeline_consolidation = (
            self._temporal_pipeline is not None
        )
        if self._cycle % self.CONSOLIDATION_INTERVAL_CYCLES == 0:
            if not use_pipeline_consolidation:
                consolidation = await asyncio.to_thread(self.memory.consolidate)
                self._emit("consolidation", {
                    "pruned": consolidation.get("pruned", 0),
                    "crystallized": consolidation.get("crystallized", 0),
                    "decayed": consolidation.get("decayed", 0),
                })
            else:
                self._emit("consolidation", {
                    "message": "Delegated to temporal pipeline (incremental)",
                })

        if self._hei and self._cycle % 50 == 0:
            try:
                ids, vecs = await asyncio.to_thread(
                    self.memory.store.export_all_vectors)
                if len(ids) > 50:
                    await asyncio.to_thread(self._hei.build, ids, vecs)
                    self._emit("hei_rebuild", {
                        "concepts": len(ids),
                        "stats": self._hei.stats(),
                    })
            except Exception as exc:
                log.debug("HEI rebuild error: %s", exc)

        self_play_result = {"action": "skip"}
        if self._cycle % self.SELF_PLAY_INTERVAL_CYCLES == 0:
            self_play_result = await asyncio.to_thread(
                self.memory.self_play_cycle)
            if self_play_result.get("action") != "skip":
                self._emit("self_play", {
                    "action": self_play_result.get("action"),
                    "concept": self_play_result.get("concept", ""),
                })

        store_stats = self.memory.store.stats()

        metrics = TrainingMetrics(
            cycle=self._cycle,
            phase=phase.value,
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

        self._emit("cycle_end", {
            "phase": phase.value,
            "concepts": metrics.total_concepts,
            "associations": metrics.total_associations,
            "learned": metrics.concepts_this_cycle,
            "images": metrics.images_this_cycle,
            "duration_s": round(metrics.cycle_duration_s, 2),
        })

        self._save_state()

        return metrics

    def _log_metrics(self, metrics: TrainingMetrics) -> None:
        if self._cycle % self.METRICS_LOG_INTERVAL_CYCLES == 0:
            log.info(
                "Cycle %d [%s]: concepts=%d assoc=%d learned=%d imgs=%d "
                "pruned=%d crystal=%d dur=%.1fs uptime=%.0fs",
                metrics.cycle, metrics.phase, metrics.total_concepts,
                metrics.total_associations, metrics.concepts_this_cycle,
                metrics.images_this_cycle, metrics.consolidation_pruned,
                metrics.consolidation_crystallized,
                metrics.cycle_duration_s, metrics.uptime_s,
            )

        if self.training_db:
            try:
                self.training_db.log_training_cycle({
                    "cycle": metrics.cycle,
                    "phase": metrics.phase,
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
            "phase": m.phase,
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

    def run_benchmark(self) -> dict:
        """Run the 20 benchmark questions and score the AI's knowledge.

        Returns a score 0-100 and per-question results.
        """
        results = []
        correct = 0

        for bm in BENCHMARK_QUESTIONS:
            collapse = self.memory.query(bm["question"], n_results=5)

            if collapse.confidence > 0.3 and collapse.context:
                try:
                    answer = self.memory.llm.answer_with_context(
                        bm["question"], collapse.context, max_new_tokens=200,
                    )
                except Exception:
                    answer = ""
            else:
                answer = ""

            expected_lower = bm["answer"].lower()
            answer_lower = answer.lower()

            key_terms = [t.strip() for t in expected_lower.split()
                         if len(t.strip()) > 2]
            matched = sum(1 for t in key_terms if t in answer_lower)
            match_ratio = matched / max(len(key_terms), 1)

            is_correct = match_ratio >= 0.5

            if is_correct:
                correct += 1

            results.append({
                "id": bm["id"],
                "question": bm["question"],
                "expected": bm["answer"],
                "answer": answer[:300],
                "confidence": round(collapse.confidence, 3),
                "match_ratio": round(match_ratio, 3),
                "correct": is_correct,
                "category": bm["category"],
            })

        score = round((correct / len(BENCHMARK_QUESTIONS)) * 100, 1)
        summary = {
            "score": score,
            "correct": correct,
            "total": len(BENCHMARK_QUESTIONS),
            "results": results,
            "timestamp": time.time(),
        }
        log.info("Benchmark score: %s/%s (%s%%)",
                 correct, len(BENCHMARK_QUESTIONS), score)
        return summary

    def status(self) -> dict:
        result = {
            "running": self._running,
            "cycle": self._cycle,
            "phase": self.current_phase().value,
            "uptime_s": round(time.time() - self._start_time, 1) if self._start_time else 0,
            "latest_metrics": self.latest_metrics(),
            "memory": self.memory.stats(),
            "agents": self.agent_manager.stats(),
        }
        if self._temporal_pipeline:
            result["pipeline"] = self._temporal_pipeline.stats()
        if self._hei:
            result["hei"] = self._hei.stats()
        return result

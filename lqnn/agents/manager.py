"""Agent manager -- orchestrates crawling, analysis, judging, and learning."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from lqnn.agents.browser_agent import BrowserAgent
from lqnn.core.associative_memory import AssociativeMemory

log = logging.getLogger(__name__)


@dataclass
class KnowledgeGap:
    concept: str
    confidence: float
    priority: float
    created_at: float = field(default_factory=time.time)


@dataclass
class CycleReport:
    cycle: int
    concepts_learned: int
    images_processed: int
    gaps_resolved: int
    duration_s: float
    timestamp: float = field(default_factory=time.time)


class JudgeAgent:
    """Validates knowledge quality before it enters the memory.

    Checks:
    - Text length and coherence (not empty, not gibberish)
    - Image validity (can be decoded by PIL)
    - Duplication (concept already exists with high confidence)
    """

    MIN_TEXT_LENGTH = 20
    MAX_TEXT_LENGTH = 5000
    DUPLICATE_THRESHOLD = 0.12  # cosine distance below this = duplicate

    def judge_text(self, text: str) -> tuple[bool, str]:
        if not text or len(text.strip()) < self.MIN_TEXT_LENGTH:
            return False, "text_too_short"
        if len(text) > self.MAX_TEXT_LENGTH:
            text = text[:self.MAX_TEXT_LENGTH]
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.5:
            return False, "text_not_coherent"
        return True, "ok"

    def judge_image(self, image_data: bytes) -> tuple[bool, str]:
        if len(image_data) < 1000:
            return False, "image_too_small"
        try:
            from PIL import Image
            from io import BytesIO
            img = Image.open(BytesIO(image_data))
            w, h = img.size
            if w < 50 or h < 50:
                return False, "image_too_small_dims"
            return True, "ok"
        except Exception:
            return False, "image_invalid"

    def is_duplicate(self, memory: AssociativeMemory,
                     vector: np.ndarray) -> bool:
        results = memory.store.query_concepts(vector, n=1)
        if results and results[0].get("distance", 1.0) < self.DUPLICATE_THRESHOLD:
            return True
        return False


class AgentManager:
    """Orchestrates the autonomous knowledge acquisition pipeline.

    Flow:
    1. Detect knowledge gaps (low-confidence areas in memory)
    2. Send browser agent to search the web
    3. Judge the results for quality
    4. Feed approved content into the associative memory
    """

    def __init__(self, memory: AssociativeMemory,
                 browser: BrowserAgent | None = None) -> None:
        self.memory = memory
        self.browser = browser or BrowserAgent()
        self.judge = JudgeAgent()
        self._cycle = 0
        self._reports: list[CycleReport] = []
        self._gap_queue: list[KnowledgeGap] = []
        self._online = True

    def set_online(self, online: bool) -> None:
        self._online = online

    def detect_knowledge_gaps(self) -> list[KnowledgeGap]:
        """Find concepts with low confidence or sparse associations."""
        gaps = []
        volatile = self.memory.store.get_volatile_concepts(threshold=0.7)
        for v in volatile[:10]:
            meta = v.get("metadata", {})
            doc = v.get("document", "")
            if doc:
                gaps.append(KnowledgeGap(
                    concept=doc,
                    confidence=meta.get("confidence", 0.5),
                    priority=meta.get("volatility", 0.8),
                ))

        if not gaps and self.memory.store.concept_count() < 50:
            seed_concepts = [
                "technology", "nature", "science", "mathematics",
                "language", "music", "food", "geography",
            ]
            for sc in seed_concepts:
                existing = self.memory.store.get_concept(
                    self.memory._make_id(sc))
                if not existing:
                    gaps.append(KnowledgeGap(
                        concept=sc, confidence=0.0, priority=1.0,
                    ))
                    break

        gaps.sort(key=lambda g: g.priority, reverse=True)
        self._gap_queue = gaps
        return gaps

    async def run_cycle(self) -> CycleReport:
        """Run one complete knowledge acquisition cycle."""
        t0 = time.time()
        self._cycle += 1
        concepts_learned = 0
        images_processed = 0
        gaps_resolved = 0

        if not self._gap_queue:
            self.detect_knowledge_gaps()

        gaps_to_process = self._gap_queue[:3]
        self._gap_queue = self._gap_queue[3:]

        for gap in gaps_to_process:
            if self._online:
                try:
                    learned, imgs = await self._process_gap_online(gap)
                    concepts_learned += learned
                    images_processed += imgs
                    gaps_resolved += 1
                except Exception as exc:
                    log.warning("Gap processing failed for '%s': %s",
                                gap.concept, exc)
            else:
                self._process_gap_offline(gap)
                gaps_resolved += 1

        report = CycleReport(
            cycle=self._cycle,
            concepts_learned=concepts_learned,
            images_processed=images_processed,
            gaps_resolved=gaps_resolved,
            duration_s=time.time() - t0,
        )
        self._reports.append(report)
        if len(self._reports) > 100:
            self._reports = self._reports[-50:]
        return report

    async def _process_gap_online(self, gap: KnowledgeGap) -> tuple[int, int]:
        """Search the web for a concept and learn from results."""
        learned = 0
        imgs = 0

        search_result = await self.browser.search(gap.concept)
        if not search_result.success:
            self.memory.learn_concept(gap.concept, source="seed")
            return 1, 0

        for r in search_result.results[:2]:
            page = await self.browser.fetch_page(r["url"], download_images=True)
            if not page.success:
                continue

            ok, reason = self.judge.judge_text(page.text)
            if ok:
                sentences = [s.strip() for s in page.text.split(".")
                             if len(s.strip()) > 20]
                for sentence in sentences[:5]:
                    concept_text = sentence[:100]
                    vec = self.memory.clip.encode_text(concept_text)
                    if not self.judge.is_duplicate(self.memory, vec):
                        self.memory.learn_concept(
                            concept_text, source=page.url,
                        )
                        learned += 1

            for img_data in page.images:
                ok, reason = self.judge.judge_image(img_data)
                if ok:
                    vec = self.memory.clip.encode_image(img_data)
                    if not self.judge.is_duplicate(self.memory, vec):
                        self.memory.learn_from_image(
                            img_data, source_url=page.url,
                        )
                        imgs += 1

        return learned, imgs

    def _process_gap_offline(self, gap: KnowledgeGap) -> None:
        """Reinforce a gap using existing knowledge (self-play)."""
        self.memory.self_play_cycle()

    async def request_search(self, query: str) -> CycleReport:
        """Manually triggered search from the UI."""
        self._gap_queue.insert(0, KnowledgeGap(
            concept=query, confidence=0.0, priority=1.0,
        ))
        return await self.run_cycle()

    def stats(self) -> dict:
        return {
            "cycle": self._cycle,
            "online": self._online,
            "gap_queue_size": len(self._gap_queue),
            "total_reports": len(self._reports),
            "last_report": (
                {
                    "cycle": self._reports[-1].cycle,
                    "concepts_learned": self._reports[-1].concepts_learned,
                    "images_processed": self._reports[-1].images_processed,
                    "duration_s": round(self._reports[-1].duration_s, 2),
                }
                if self._reports else None
            ),
        }

    async def shutdown(self) -> None:
        await self.browser.close()

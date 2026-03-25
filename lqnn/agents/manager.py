"""Agent manager -- orchestrates crawling, analysis, judging, and learning."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import numpy as np

from lqnn.agents.browser_agent import BrowserAgent
from lqnn.core.associative_memory import AssociativeMemory

log = logging.getLogger(__name__)

BOILERPLATE_PATTERNS = [
    "log in", "sign up", "sign in", "cookie", "subscribe",
    "newsletter", "privacy policy", "terms of service",
    "click here", "read more", "see the announcement",
    "accept all", "manage preferences", "advertisement",
    "please enable javascript", "your browser does not support",
    "all rights reserved", "powered by", "skip to main content",
    "skip to content", "toggle navigation", "close menu",
    "share on facebook", "share on twitter", "follow us",
    "back to top", "load more", "show more", "view all",
    "add to cart", "buy now", "free trial", "download now",
    "unsubscribe", "opt out", "manage consent", "we use cookies",
    "this site uses", "by continuing", "you agree to",
    "enable notifications", "allow notifications", "push notifications",
    "sponsored content", "related articles", "trending now",
    "sign up for free", "create account", "forgot password",
    "leave a comment", "post a comment", "reply to",
    "breaking news", "latest news", "top stories",
    "page not found", "404 error", "access denied",
    "captcha", "verify you are human", "cloudflare",
    "%n%n", "%s%s", "rnd=", ";app=", ";uid=",
]

BOILERPLATE_RE = re.compile(
    r"<script|</script>|<style|</style>|alert\s*\(|function\s*\(|"
    r"var\s+\w+\s*=|document\.|window\.|\.getElementById|\.querySelector|"
    r"\{[^}]*:[^}]*\}|https?://\S{60,}",
    re.IGNORECASE,
)

DOMAIN_BLACKLIST = {
    "pinterest.com", "pinterest.co", "facebook.com", "instagram.com",
    "twitter.com", "x.com", "tiktok.com", "reddit.com",
    "amazon.com", "ebay.com", "aliexpress.com",
    "popads.net", "adclick.com",
}

DOMAIN_WHITELIST_PRIORITY = {
    "wikipedia.org", "britannica.com", "arxiv.org", "nature.com",
    "sciencedirect.com", "github.com", "stackoverflow.com",
    "w3schools.com", "mdn.mozilla.org", "docs.python.org",
    "khanacademy.org", "mit.edu", "stanford.edu",
}


@dataclass
class KnowledgeGap:
    concept: str
    confidence: float
    priority: float
    prefer_images: bool = False
    created_at: float = field(default_factory=time.time)


@dataclass
class CycleReport:
    cycle: int
    concepts_learned: int
    images_processed: int
    gaps_resolved: int
    duration_s: float
    phase: str = ""
    timestamp: float = field(default_factory=time.time)


class JudgeAgent:
    """Extreme quality gate for web-sourced knowledge.

    v5: Much stricter filtering to prevent garbage from entering the brain.
    Multi-layer quality gate:
    1. Boilerplate blacklist -- reject web junk (login prompts, cookie banners)
    2. Alpha ratio -- reject code, URLs, garbled text (raised to 0.70)
    3. Decoherence shield -- reject binary artefacts and corrupted data
    4. CLIP relevance -- reject content unrelated to search topic (raised to 0.40)
    5. Minimum word count -- reject fragments that are too sparse
    6. Duplicate detection -- reject near-identical vectors (tightened to 0.10)
    """

    MIN_TEXT_LENGTH = 50
    MAX_TEXT_LENGTH = 3000
    MIN_WORD_COUNT = 8
    DUPLICATE_THRESHOLD = 0.10
    CLIP_RELEVANCE_THRESHOLD = 0.40
    TEXT_RELEVANCE_THRESHOLD = 0.35

    @staticmethod
    def is_boilerplate(text: str) -> bool:
        """Check if text matches common web boilerplate patterns."""
        lower = text.lower()
        hits = sum(1 for p in BOILERPLATE_PATTERNS if p in lower)
        if hits >= 2:
            return True
        if BOILERPLATE_RE.search(text):
            return True
        return False

    def judge_text(self, text: str) -> tuple[bool, str]:
        if not text or len(text.strip()) < self.MIN_TEXT_LENGTH:
            return False, "text_too_short"
        if len(text) > self.MAX_TEXT_LENGTH:
            text = text[:self.MAX_TEXT_LENGTH]
        if self.is_boilerplate(text):
            return False, "boilerplate"
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.70:
            return False, "text_not_coherent"
        word_count = len(text.split())
        if word_count < self.MIN_WORD_COUNT:
            return False, "too_few_words"
        if not self._passes_decoherence_shield(text):
            return False, "decoherence_garbage"
        return True, "ok"

    @staticmethod
    def _passes_decoherence_shield(text: str) -> bool:
        """Reject corrupted data, binary artefacts, encoded strings."""
        sample = text[:300].lower()
        garbage_indicators = [
            "%n%n", "%s%s", "rnd=", ";app=", ";uid=",
            "\\x", "\\u00", "68bac", "0a27c", "itrade",
            "76ff3d", "b8fca4", "r_1;uid",
        ]
        hits = sum(1 for g in garbage_indicators if g in sample)
        if hits >= 1:
            return False
        non_ascii = sum(1 for c in text[:200] if ord(c) > 127 and not c.isalpha())
        if non_ascii / max(len(text[:200]), 1) > 0.15:
            return False
        return True

    def judge_text_relevance(self, memory: AssociativeMemory,
                             text: str, concept: str) -> tuple[bool, float]:
        """Check CLIP cosine similarity between text and the search concept."""
        try:
            text_vec = memory.clip.encode_text(text[:512])
            concept_vec = memory.clip.encode_text(concept)
            similarity = float(np.dot(text_vec, concept_vec))
            return similarity >= self.TEXT_RELEVANCE_THRESHOLD, similarity
        except Exception:
            return False, 0.0

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
            if w < 100 and h < 100:
                return False, "likely_icon"
            return True, "ok"
        except Exception:
            return False, "image_invalid"

    def judge_image_relevance(self, memory: AssociativeMemory,
                              image_data: bytes, concept: str) -> tuple[bool, float]:
        """Check if an image is relevant to a concept using CLIP similarity."""
        try:
            img_vec = memory.clip.encode_image(image_data)
            text_vec = memory.clip.encode_text(concept)
            similarity = float(np.dot(img_vec, text_vec))
            return similarity > self.CLIP_RELEVANCE_THRESHOLD, similarity
        except Exception:
            return False, 0.0

    def is_duplicate(self, memory: AssociativeMemory,
                     vector: np.ndarray) -> bool:
        results = memory.store.query_concepts(vector, n=1)
        if results and results[0].get("distance", 1.0) < self.DUPLICATE_THRESHOLD:
            return True
        return False

    @staticmethod
    def is_blacklisted_domain(url: str) -> bool:
        try:
            domain = urlparse(url).netloc.lower()
            for bl in DOMAIN_BLACKLIST:
                if bl in domain:
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def is_whitelisted_domain(url: str) -> bool:
        try:
            domain = urlparse(url).netloc.lower()
            for wl in DOMAIN_WHITELIST_PRIORITY:
                if wl in domain:
                    return True
        except Exception:
            pass
        return False


class AgentManager:
    """On-demand knowledge acquisition agent.

    v5: Agents are NOT called from the training loop.  They activate ONLY:
      1. When a user chat has very low confidence (reactive search)
      2. When the user explicitly triggers a search from the UI

    Every piece of web content passes through extreme quality filtering
    (JudgeAgent + Decoherence Shield) before entering the knowledge base.
    """

    def __init__(self, memory: AssociativeMemory,
                 browser: BrowserAgent | None = None) -> None:
        self.memory = memory
        self.browser = browser or BrowserAgent()
        self.judge = JudgeAgent()
        self.smart_search = None
        self._cycle = 0
        self._reports: list[CycleReport] = []
        self._gap_queue: list[KnowledgeGap] = []
        self._online = True
        self._phase = None
        self._event_callback = None
        self._activity_log: list[dict] = []

        try:
            from lqnn.agents.smart_search import SmartSearchAgent
            self.smart_search = SmartSearchAgent(browser=self.browser)
        except Exception:
            log.warning("SmartSearchAgent not available, using basic browser")

    def set_event_callback(self, callback) -> None:
        self._event_callback = callback

    def _emit(self, event_type: str, data: dict) -> None:
        entry = {
            "type": event_type,
            "timestamp": time.time(),
            **data,
        }
        self._activity_log.append(entry)
        if len(self._activity_log) > 500:
            self._activity_log = self._activity_log[-250:]
        if self._event_callback:
            try:
                self._event_callback(entry)
            except Exception:
                pass

    @property
    def activity_log(self) -> list[dict]:
        return self._activity_log

    def set_online(self, online: bool) -> None:
        self._online = online

    def set_phase(self, phase) -> None:
        self._phase = phase

    def detect_knowledge_gaps(self) -> list[KnowledgeGap]:
        """Find concepts with low confidence or sparse associations.

        Phase-aware: uses different seed lists depending on training phase.
        Also derives seeds from user-curated knowledge library uploads.
        """
        from lqnn.training.continuous_trainer import (
            TrainingPhase, VISUAL_SEEDS, ABSTRACT_SEEDS, BENCHMARK_SEEDS,
            FRONTIER_BENCHMARK_SEEDS,
        )

        gaps = []

        for bq in BENCHMARK_SEEDS:
            existing = self.memory.store.get_concept(
                self.memory._make_id(bq))
            if not existing:
                gaps.append(KnowledgeGap(
                    concept=bq, confidence=0.0, priority=1.5,
                    prefer_images=False,
                ))
                if len(gaps) >= 3:
                    break

        if len(gaps) < 3:
            for fq in FRONTIER_BENCHMARK_SEEDS[:4]:
                existing = self.memory.store.get_concept(
                    self.memory._make_id(fq))
                if not existing:
                    gaps.append(KnowledgeGap(
                        concept=fq, confidence=0.0, priority=1.35,
                        prefer_images=False,
                    ))
                if len(gaps) >= 3:
                    break

        volatile = self.memory.store.get_volatile_concepts(threshold=0.7)
        for v in volatile[:10]:
            meta = v.get("metadata", {})
            doc = v.get("document", "")
            if doc and not JudgeAgent.is_boilerplate(doc):
                prefer_images = self._phase == TrainingPhase.VISUAL_OBJECTS
                gaps.append(KnowledgeGap(
                    concept=doc,
                    confidence=meta.get("confidence", 0.5),
                    priority=meta.get("volatility", 0.8),
                    prefer_images=prefer_images,
                ))

        kb_seeds = self._derive_seeds_from_library()
        for sc in kb_seeds:
            existing = self.memory.store.get_concept(
                self.memory._make_id(sc))
            if not existing:
                gaps.append(KnowledgeGap(
                    concept=sc, confidence=0.0, priority=1.2,
                    prefer_images=False,
                ))
                if len(gaps) >= 5:
                    break

        if len(gaps) < 3:
            if self._phase == TrainingPhase.VISUAL_OBJECTS:
                seeds = VISUAL_SEEDS
            elif self._phase == TrainingPhase.ABSTRACT_CONCEPTS:
                seeds = ABSTRACT_SEEDS
            else:
                seeds = self._derive_seeds_from_knowledge()

            for sc in seeds:
                existing = self.memory.store.get_concept(
                    self.memory._make_id(sc))
                if not existing:
                    prefer_images = self._phase == TrainingPhase.VISUAL_OBJECTS
                    gaps.append(KnowledgeGap(
                        concept=sc, confidence=0.0, priority=1.0,
                        prefer_images=prefer_images,
                    ))
                    if len(gaps) >= 5:
                        break

        gaps.sort(key=lambda g: g.priority, reverse=True)
        self._gap_queue = gaps
        return gaps

    def _derive_seeds_from_library(self) -> list[str]:
        """Extract key topics from user-curated knowledge base content."""
        try:
            all_data = self.memory.store._concepts.get(
                include=["documents", "metadatas"],
                limit=100,
            )
            if not all_data["ids"]:
                return []

            curated_texts = []
            for i, meta in enumerate(all_data["metadatas"]):
                is_curated = (
                    meta.get("curation") == "user_curated"
                    or str(meta.get("source", "")).startswith("user_curated:")
                )
                if is_curated:
                    full_text = meta.get("full_text", "")
                    if full_text and len(full_text) > 50:
                        curated_texts.append(full_text[:500])

            if not curated_texts:
                return []

            combined = "\n".join(curated_texts[:5])
            raw = self.memory.llm.generate(
                f"From this knowledge base content, extract 5 key topics or "
                f"concepts that should be studied further. Output ONLY a "
                f"numbered list, one concept per line (2-5 words each).\n\n"
                f"Content:\n{combined[:1500]}",
                max_new_tokens=150,
                temperature=0.5,
            )

            import re
            seeds = []
            for line in raw.strip().splitlines():
                cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
                if cleaned and 3 <= len(cleaned) <= 60:
                    seeds.append(cleaned.lower())
            return seeds[:5]

        except Exception as e:
            log.debug("Failed to derive library seeds: %s", e)
            return []

    def _derive_seeds_from_knowledge(self) -> list[str]:
        """In self-evolution phase, derive new topics from existing associations."""
        all_data = self.memory.store._associations.get(
            include=["documents"],
            limit=200,
        )
        if not all_data["documents"]:
            return ["philosophy", "creativity", "universe"]

        docs = [d for d in all_data["documents"] if d and len(d) > 2]
        if not docs:
            return ["philosophy", "creativity", "universe"]

        rng = np.random.default_rng()
        chosen = rng.choice(docs, size=min(10, len(docs)), replace=False)
        return list(chosen)

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
                    self._emit("agent_error", {
                        "concept": gap.concept,
                        "error": str(exc),
                    })
            else:
                self._process_gap_offline(gap)
                gaps_resolved += 1

        phase_name = self._phase.value if self._phase else "unknown"
        report = CycleReport(
            cycle=self._cycle,
            concepts_learned=concepts_learned,
            images_processed=images_processed,
            gaps_resolved=gaps_resolved,
            duration_s=time.time() - t0,
            phase=phase_name,
        )
        self._reports.append(report)
        if len(self._reports) > 100:
            self._reports = self._reports[-50:]
        return report

    async def _process_gap_online(self, gap: KnowledgeGap) -> tuple[int, int]:
        """Search the web for a concept with extreme quality filtering.

        Pipeline per sentence:
        1. Decoherence shield (binary/garbage rejection)
        2. Boilerplate blacklist check
        3. Alpha ratio + word count validation
        4. CLIP relevance gate (raised threshold: 0.40)
        5. LLM extracts a clean 2-5 word concept label
        6. Duplicate check (tightened threshold: 0.10)
        7. Store only if ALL gates pass

        Respects GPU exclusion gate -- skips LLM work during user chat.
        """
        import asyncio
        learned = 0
        imgs = 0

        search_query = gap.concept
        if gap.prefer_images:
            search_query = f"{gap.concept} photo image"

        self._emit("search", {
            "concept": gap.concept,
            "query": search_query,
        })

        if hasattr(self, 'smart_search') and self.smart_search is not None:
            search_result = await self.smart_search.search(gap.concept)
        else:
            search_result = await self.browser.search(search_query)

        if not search_result.success:
            state = await asyncio.to_thread(
                self.memory.learn_concept, gap.concept, None, "seed")
            self._emit("learn", {
                "concept": gap.concept,
                "source": "seed",
                "associations": len(state.associations),
            })
            return 1, 0

        results = search_result.results[:5]
        whitelisted = [r for r in results
                       if self.judge.is_whitelisted_domain(r["url"])]
        others = [r for r in results
                  if not self.judge.is_whitelisted_domain(r["url"])
                  and not self.judge.is_blacklisted_domain(r["url"])]
        sorted_results = (whitelisted + others)[:3]

        for r in sorted_results:
            if self.judge.is_blacklisted_domain(r["url"]):
                self._emit("judge_reject", {
                    "concept": gap.concept,
                    "reason": f"blacklisted_domain: {r['url'][:60]}",
                })
                continue

            page = await self.browser.fetch_page(
                r["url"], download_images=gap.prefer_images,
            )
            if not page.success:
                continue

            ok, reason = self.judge.judge_text(page.text)
            if ok:
                sentences = self._extract_quality_sentences(page.text)

                for sentence in sentences[:8]:
                    if self.judge.is_boilerplate(sentence):
                        continue

                    relevant, sim = await asyncio.to_thread(
                        self.judge.judge_text_relevance,
                        self.memory, sentence, gap.concept,
                    )
                    if not relevant:
                        self._emit("judge_reject", {
                            "concept": gap.concept,
                            "reason": f"text_not_relevant (sim={sim:.3f})",
                        })
                        continue

                    concept_label = await asyncio.to_thread(
                        self._extract_concept_label, sentence, gap.concept,
                    )
                    if not concept_label or len(concept_label) < 3:
                        continue

                    vec = await asyncio.to_thread(
                        self.memory.clip.encode_text, concept_label)
                    if self.judge.is_duplicate(self.memory, vec):
                        continue

                    source_domain = urlparse(page.url).netloc
                    state = await asyncio.to_thread(
                        self.memory.learn_concept,
                        concept_label, None, page.url,
                    )
                    learned += 1
                    self._emit("learn", {
                        "concept": concept_label[:60],
                        "source": "web",
                        "domain": source_domain,
                        "relevance": round(sim, 3),
                        "associations": len(state.associations),
                    })

            for img_data in page.images[:5]:
                ok, reason = self.judge.judge_image(img_data)
                if not ok:
                    continue

                relevant, score = await asyncio.to_thread(
                    self.judge.judge_image_relevance,
                    self.memory, img_data, gap.concept,
                )
                if not relevant:
                    self._emit("judge_reject", {
                        "concept": gap.concept,
                        "reason": f"image_not_relevant (score={score:.2f})",
                    })
                    continue

                vec = await asyncio.to_thread(
                    self.memory.clip.encode_image, img_data)
                if not self.judge.is_duplicate(self.memory, vec):
                    result = await asyncio.to_thread(
                        self.memory.learn_from_image,
                        img_data, page.url,
                    )
                    if result:
                        imgs += 1
                        self._emit("learn_image", {
                            "concept": result.concept,
                            "relevance_score": round(score, 3),
                            "associations": len(result.associations),
                        })

        return learned, imgs

    @staticmethod
    def _extract_quality_sentences(text: str) -> list[str]:
        """Extract meaningful sentences from page text with strict filtering."""
        sentences = []
        for raw in re.split(r'[.!?]\s+', text):
            s = raw.strip()
            if len(s) < 60:
                continue
            if len(s) > 300:
                s = s[:300]
            alpha_count = sum(c.isalpha() or c.isspace() for c in s)
            if alpha_count / max(len(s), 1) < 0.75:
                continue
            if s.count("http") > 0 or s.count("@") > 0:
                continue
            word_count = len(s.split())
            if word_count < 8:
                continue
            if not JudgeAgent._passes_decoherence_shield(s):
                continue
            sentences.append(s)
        return sentences

    def _extract_concept_label(self, sentence: str, topic: str) -> str:
        """Use LLM to extract a clean concept label from a sentence."""
        try:
            label = self.memory.llm.generate(
                f'Given the topic "{topic}", extract the main concept from '
                f'this text as a short phrase (2-8 words). Reply with ONLY '
                f'the concept, nothing else.\n\nText: "{sentence[:250]}"',
                max_new_tokens=25,
                temperature=0.1,
            )
            label = label.strip().strip('"').strip("'").strip(".")
            label = re.sub(r'^(the |a |an )', '', label, flags=re.IGNORECASE)
            if len(label) < 3 or len(label) > 80:
                return ""
            if JudgeAgent.is_boilerplate(label):
                return ""
            return label.lower()
        except Exception:
            return sentence[:60].strip().lower()

    def _process_gap_offline(self, gap: KnowledgeGap) -> None:
        """Reinforce a gap using existing knowledge (self-play)."""
        self.memory.self_play_cycle()

    async def request_search(self, query: str) -> CycleReport:
        """Manually triggered search from the UI."""
        self._gap_queue.insert(0, KnowledgeGap(
            concept=query, confidence=0.0, priority=1.0,
            prefer_images=True,
        ))
        return await self.run_cycle()

    def stats(self) -> dict:
        return {
            "cycle": self._cycle,
            "online": self._online,
            "phase": self._phase.value if self._phase else "unknown",
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
        if self.smart_search:
            await self.smart_search.close()

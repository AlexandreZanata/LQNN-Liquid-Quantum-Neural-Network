"""Quantum Associative Memory -- the brain of the LQNN system.

Implements:
- Superposition: a concept exists as a cloud of related vectors simultaneously
- Collapse: on query, the system collapses to the most relevant associations
- Volatility decay: frequently accessed vectors become stable (crystallize),
  unused ones decay and are eventually pruned
- Consolidation: periodic cycles that promote short-term to long-term memory
- Visual-first learning: images weighted 70/30 over text (human brain is ~80% visual)
- Multi-image learning: average multiple image vectors for robust representations
- Network crystallization: concepts with many interconnections stabilize faster
"""

from __future__ import annotations

import hashlib
import logging
import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from lqnn.core.vector_store import VectorEntry, VectorStore
from lqnn.models.clip_encoder import CLIPEncoder
from lqnn.models.llm_engine import LLMEngine

log = logging.getLogger(__name__)

IMAGE_WEIGHT = 0.7
TEXT_WEIGHT = 0.3
EMBED_CACHE_SIZE = 2048


@dataclass
class QuantumState:
    """A concept in superposition with its associations."""
    concept: str
    primary_vector: np.ndarray
    associations: list[dict] = field(default_factory=list)
    volatility: float = 1.0
    confidence: float = 0.5
    access_count: int = 0


@dataclass
class CollapseResult:
    """Result of collapsing a query against the memory."""
    query: str
    matched_concepts: list[dict]
    confidence: float
    context: str
    associations: list[dict]


class AssociativeMemory:
    """The quantum associative brain.

    Learns like a human: see 'banana' -> encode visually -> generate
    associations (yellow, sweet, fruit, monkey...) -> store all vectors
    in superposition -> collapse to answer queries.
    """

    VOLATILITY_DECAY_RATE = 0.05
    VOLATILITY_INCREASE_RATE = 0.02
    PRUNE_THRESHOLD = 0.95
    CRYSTALLIZE_THRESHOLD = 0.2
    MIN_CONFIDENCE_TO_ANSWER = 0.3
    NETWORK_CRYSTALLIZATION_BONUS = 0.01
    USER_CURATED_VOLATILITY_BONUS = 0.02

    def __init__(self, store: VectorStore, clip: CLIPEncoder,
                 llm: LLMEngine) -> None:
        self.store = store
        self.clip = clip
        self.llm = llm
        self._learn_count = 0
        self._query_count = 0
        self._embed_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._assoc_bg_queue: queue.Queue = queue.Queue(maxsize=500)
        self._assoc_bg_thread: threading.Thread | None = None
        self._assoc_bg_running = False
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._start_bg_association_worker()

    def _cached_encode_text(self, text: str) -> np.ndarray:
        """CLIP text encoding with LRU cache."""
        key = text[:512]
        if key in self._embed_cache:
            self._embed_cache.move_to_end(key)
            return self._embed_cache[key]
        vec = self.clip.encode_text(key)
        self._embed_cache[key] = vec
        if len(self._embed_cache) > EMBED_CACHE_SIZE:
            self._embed_cache.popitem(last=False)
        return vec

    def _start_bg_association_worker(self) -> None:
        """Background thread that processes deferred association generation."""
        if self._assoc_bg_thread and self._assoc_bg_thread.is_alive():
            return
        self._assoc_bg_running = True
        self._assoc_bg_thread = threading.Thread(
            target=self._bg_association_loop, daemon=True,
            name="assoc-bg-worker",
        )
        self._assoc_bg_thread.start()

    def _bg_association_loop(self) -> None:
        while self._assoc_bg_running:
            try:
                concept, primary_vec, n = self._assoc_bg_queue.get(timeout=2.0)
                self._generate_associations_sync(concept, primary_vec, n)
            except queue.Empty:
                continue
            except Exception as e:
                log.debug("BG association error: %s", e)

    # -- Learning --

    def learn_concept(self, concept: str, image: bytes | None = None,
                      source: str = "manual",
                      initial_confidence: float = 0.5) -> QuantumState:
        """Learn a concept by encoding it and generating associations.

        Visual-first learning: when an image is available, the vector is
        weighted 70% image / 30% text, mirroring how the human brain
        processes ~80% of information visually.
        """
        concept_lower = concept.strip().lower()
        concept_id = self._make_id(concept_lower)

        text_vec = self._cached_encode_text(concept_lower)

        if image is not None:
            img_vec = self.clip.encode_image(image)
            primary_vec = IMAGE_WEIGHT * img_vec + TEXT_WEIGHT * text_vec
            primary_vec = primary_vec / np.linalg.norm(primary_vec)
        else:
            primary_vec = text_vec

        entry = VectorEntry(
            id=concept_id,
            vector=primary_vec,
            concept=concept_lower,
            source=source,
            volatility=1.0,
            confidence=initial_confidence,
        )
        self.store.add_concept(entry)

        immediate = self._generate_associations_sync(
            concept_lower, primary_vec, n=5)

        try:
            self._assoc_bg_queue.put_nowait(
                (concept_lower, primary_vec, 25))
        except queue.Full:
            pass

        self._learn_count += 1
        log.info(
            "Learned '%s' with %d immediate associations (source=%s)",
            concept_lower, len(immediate), source,
        )
        return QuantumState(
            concept=concept_lower,
            primary_vector=primary_vec,
            associations=immediate,
            volatility=1.0,
            confidence=initial_confidence,
        )

    def learn_concept_multi_image(self, concept: str,
                                  images: list[bytes],
                                  source: str = "multi_image") -> QuantumState:
        """Learn from multiple images of the same concept.

        Averages all image vectors with the text vector for a more robust
        representation, like seeing a banana from many angles and contexts.
        """
        concept_lower = concept.strip().lower()
        concept_id = self._make_id(concept_lower)

        text_vec = self._cached_encode_text(concept_lower)

        img_vecs = []
        for img in images:
            try:
                v = self.clip.encode_image(img)
                img_vecs.append(v)
            except Exception:
                continue

        if img_vecs:
            avg_img_vec = np.mean(img_vecs, axis=0)
            avg_img_vec = avg_img_vec / np.linalg.norm(avg_img_vec)
            primary_vec = IMAGE_WEIGHT * avg_img_vec + TEXT_WEIGHT * text_vec
            primary_vec = primary_vec / np.linalg.norm(primary_vec)
        else:
            primary_vec = text_vec

        confidence = min(0.8, 0.5 + 0.05 * len(img_vecs))

        entry = VectorEntry(
            id=concept_id,
            vector=primary_vec,
            concept=concept_lower,
            source=source,
            volatility=0.8,
            confidence=confidence,
        )
        self.store.add_concept(entry)

        immediate = self._generate_associations_sync(
            concept_lower, primary_vec, n=5)
        try:
            self._assoc_bg_queue.put_nowait(
                (concept_lower, primary_vec, 25))
        except queue.Full:
            pass

        self._learn_count += 1
        log.info(
            "Learned '%s' from %d images with %d immediate associations",
            concept_lower, len(img_vecs), len(immediate),
        )
        return QuantumState(
            concept=concept_lower,
            primary_vector=primary_vec,
            associations=immediate,
            volatility=0.8,
            confidence=confidence,
        )

    def learn_from_image(self, image: bytes, source_url: str = "") -> QuantumState | None:
        """Learn from a raw image -- CLIP encodes it, LLM describes it."""
        img_vec = self.clip.encode_image(image)

        similar = self.store.query_concepts(img_vec, n=1)
        if similar and similar[0].get("distance", 1.0) < 0.15:
            log.debug("Image too similar to existing concept '%s', skipping",
                       similar[0].get("document", ""))
            return None

        description = self.llm.generate(
            "You see an image. Based on the context, what single object or "
            "concept does this image most likely represent? Reply with just "
            "the name, one or two words.",
            max_new_tokens=20, temperature=0.3,
        )
        concept = description.strip().lower().rstrip(".")

        if not concept or len(concept) > 50:
            concept = f"visual_{time.time():.0f}"

        return self.learn_concept(concept, image=image,
                                  source=source_url or "image")

    def _generate_associations(self, concept: str,
                               primary_vec: np.ndarray) -> list[dict]:
        """Full 30-association generation (used by self-play, agents)."""
        return self._generate_associations_sync(concept, primary_vec, n=30)

    def _generate_associations_sync(self, concept: str,
                                    primary_vec: np.ndarray,
                                    n: int = 30) -> list[dict]:
        """Generate and store N association vectors for a concept."""
        assoc_words = self.llm.extract_associations(concept, n=n)
        associations = []

        if assoc_words:
            assoc_vectors = self.clip.encode_texts(assoc_words)
            for word, vec in zip(assoc_words, assoc_vectors):
                strength = float(np.dot(primary_vec, vec))
                assoc_id = self.store.add_association(
                    source_concept=concept,
                    target_concept=word,
                    vector=vec,
                    strength=max(0.0, strength),
                )
                associations.append({
                    "id": assoc_id,
                    "word": word,
                    "strength": strength,
                })
        return associations

    # -- Querying (Collapse) --

    def query(self, question: str, n_results: int = 10) -> CollapseResult:
        """Collapse the quantum state to answer a question.

        Uses cached CLIP encoding and parallel Chroma queries for speed.
        """
        query_vec = self._cached_encode_text(question)

        fut_concepts = self._pool.submit(
            self.store.query_concepts, query_vec, n_results)
        fut_assocs = self._pool.submit(
            self.store.query_associations, query_vec, n_results * 2)

        concepts = fut_concepts.result()
        assoc_results = fut_assocs.result()

        self.store.batch_touch(
            [c.get("id", "") for c in concepts],
            [c.get("metadata", {}) for c in concepts],
        )

        context_parts = []
        for c in concepts[:3]:
            doc = c.get("document", "")
            meta = c.get("metadata", {})
            full_text = meta.get("full_text", "")
            dist = c.get("distance", 1.0)
            if full_text:
                context_parts.append(
                    f"- {full_text[:400]} (relevance: {1 - dist:.2f})")
            elif doc:
                context_parts.append(f"- {doc} (relevance: {1 - dist:.2f})")

        for a in assoc_results[:5]:
            doc = a.get("document", "")
            if doc:
                context_parts.append(f"- {doc}")

        context = "\n".join(context_parts) if context_parts else ""

        if concepts:
            distances = [c.get("distance", 1.0) for c in concepts[:5]]
            confidence = max(0.0, 1.0 - min(distances))
        else:
            confidence = 0.0

        self._query_count += 1
        return CollapseResult(
            query=question,
            matched_concepts=concepts,
            confidence=confidence,
            context=context,
            associations=assoc_results,
        )

    # -- Consolidation --

    def consolidate(self) -> dict:
        """Run a consolidation cycle with network-aware crystallization.

        - Decrease volatility of frequently accessed concepts (crystallize)
        - Concepts with many association links crystallize faster
        - Increase volatility of unused concepts
        - Prune concepts that exceed the volatility threshold
        """
        now = time.time()
        pruned = 0
        crystallized = 0
        decayed = 0

        all_concepts = self.store._concepts.get(
            include=["metadatas"],
        )
        if not all_concepts["ids"]:
            return {"pruned": 0, "crystallized": 0, "decayed": 0}

        assoc_stats = self.store._associations.get(
            include=["metadatas"],
        )
        concept_link_counts: dict[str, int] = {}
        if assoc_stats and assoc_stats["metadatas"]:
            for meta in assoc_stats["metadatas"]:
                src = meta.get("source_concept", "")
                if src:
                    concept_link_counts[src] = concept_link_counts.get(src, 0) + 1

        for i, cid in enumerate(all_concepts["ids"]):
            meta = all_concepts["metadatas"][i]
            volatility = meta.get("volatility", 1.0)
            access_count = meta.get("access_count", 0)
            last_accessed = meta.get("last_accessed", now)
            concept_name = meta.get("concept", "")
            age_hours = (now - last_accessed) / 3600

            link_count = concept_link_counts.get(concept_name, 0)
            network_bonus = self.NETWORK_CRYSTALLIZATION_BONUS * min(link_count, 30)

            is_curated = (
                meta.get("curation") == "user_curated"
                or str(meta.get("source", "")).startswith("user_curated:")
            )
            curation_bonus = self.USER_CURATED_VOLATILITY_BONUS if is_curated else 0.0

            if access_count > 5:
                new_vol = max(0.0, volatility - self.VOLATILITY_DECAY_RATE *
                              min(access_count, 50) - network_bonus - curation_bonus)
                if new_vol <= self.CRYSTALLIZE_THRESHOLD:
                    crystallized += 1
            elif age_hours > 24:
                decay = self.VOLATILITY_INCREASE_RATE * min(age_hours / 24, 10)
                if is_curated:
                    decay *= 0.5
                new_vol = min(1.0, volatility + decay)
                if new_vol >= self.PRUNE_THRESHOLD:
                    self.store.delete_concept(cid)
                    pruned += 1
                    continue
                decayed += 1
            else:
                new_vol = volatility

            self.store.update_metadata(cid, {"volatility": new_vol})

        result = {
            "pruned": pruned,
            "crystallized": crystallized,
            "decayed": decayed,
            "total": len(all_concepts["ids"]) - pruned,
        }
        log.info("Consolidation: %s", result)
        return result

    # -- Self-play --

    def self_play_cycle(self) -> dict:
        """Enhanced self-play with Q&A validation.

        1. Pick a random concept with stored context
        2. Generate a question about it
        3. Answer using only the knowledge base
        4. Score answer quality
        5. If quality is low, reinforce learning
        """
        count = self.store.concept_count()
        if count == 0:
            return {"action": "skip", "reason": "no_concepts"}

        all_data = self.store._concepts.get(
            include=["documents", "metadatas"],
            limit=min(count, 100),
        )
        if not all_data["documents"]:
            return {"action": "skip", "reason": "empty_documents"}

        idx = np.random.randint(len(all_data["documents"]))
        concept = all_data["documents"][idx]
        meta = all_data["metadatas"][idx] if all_data["metadatas"] else {}

        full_text = meta.get("full_text", "")

        collapse = self.query(concept)

        if full_text and len(full_text) > 50 and self.llm.ready:
            try:
                question = self.llm.generate(
                    f'Based on this knowledge: "{full_text[:500]}"\n\n'
                    f'Generate ONE specific question that tests understanding '
                    f'of the concept "{concept}". Reply with ONLY the question.',
                    max_new_tokens=60,
                    temperature=0.7,
                )
                question = question.strip()

                if question and len(question) > 10:
                    answer = self.llm.answer_with_context(
                        question, collapse.context, max_new_tokens=200,
                    )

                    quality_raw = self.llm.generate(
                        f'Rate this answer quality from 0 to 10.\n'
                        f'Question: {question}\n'
                        f'Answer: {answer[:300]}\n'
                        f'Reply with ONLY a number.',
                        max_new_tokens=5,
                        temperature=0.1,
                    )

                    import re
                    match = re.search(r"(\d+(?:\.\d+)?)", quality_raw)
                    quality = float(match.group(1)) if match else 5.0

                    if quality < 5.0 and count > 10:
                        new_assocs = self._generate_associations(
                            concept, self._cached_encode_text(concept))
                        return {
                            "action": "reinforced",
                            "concept": concept,
                            "question": question[:100],
                            "quality_score": quality,
                            "new_associations": len(new_assocs),
                            "confidence": collapse.confidence,
                        }

                    return {
                        "action": "validated",
                        "concept": concept,
                        "question": question[:100],
                        "quality_score": quality,
                        "confidence": collapse.confidence,
                    }
            except Exception as e:
                log.debug("Self-play Q&A failed: %s", e)

        if collapse.confidence < 0.5 and count > 10:
            new_assocs = self._generate_associations(
                concept, self._cached_encode_text(concept))
            return {
                "action": "reinforced",
                "concept": concept,
                "new_associations": len(new_assocs),
                "confidence_before": collapse.confidence,
            }

        return {
            "action": "validated",
            "concept": concept,
            "confidence": collapse.confidence,
        }

    # -- Cleanup --

    def cleanup_garbage(self) -> dict:
        """Scan all stored concepts and remove web boilerplate / junk.

        Returns a summary of how many concepts were purged.
        """
        from lqnn.agents.manager import JudgeAgent

        all_data = self.store._concepts.get(
            include=["documents", "metadatas"],
        )
        if not all_data["ids"]:
            return {"purged": 0, "total": 0}

        purged = 0
        for i, cid in enumerate(all_data["ids"]):
            doc = all_data["documents"][i] or ""
            meta = all_data["metadatas"][i] or {}

            is_curated = (
                meta.get("curation") == "user_curated"
                or str(meta.get("source", "")).startswith("user_curated:")
            )
            if is_curated:
                continue

            if JudgeAgent.is_boilerplate(doc):
                self.store.delete_concept(cid)
                purged += 1
                continue

            alpha_count = sum(c.isalpha() or c.isspace() for c in doc)
            if len(doc) > 0 and alpha_count / len(doc) < 0.5:
                self.store.delete_concept(cid)
                purged += 1
                continue

            if len(doc) < 10 or len(doc) > 200:
                source = str(meta.get("source", ""))
                if source not in ("seed", "manual", "user_query") and not is_curated:
                    if len(doc) > 200:
                        self.store.delete_concept(cid)
                        purged += 1
                        continue

        result = {
            "purged": purged,
            "total": len(all_data["ids"]),
            "remaining": len(all_data["ids"]) - purged,
        }
        log.info("Garbage cleanup: %s", result)
        return result

    # -- Training Questions --

    def generate_training_questions(self, concept: str, n: int = 5) -> list[str]:
        """Generate context-aware training questions from stored knowledge."""
        context_result = self.query(concept, n_results=3)
        if not context_result.context:
            return []

        raw = self.llm.generate(
            f"Based on this knowledge:\n{context_result.context}\n\n"
            f"Generate {n} specific, deep questions about '{concept}' "
            f"that test real understanding. Output as a numbered list, "
            f"one question per line.",
            max_new_tokens=300,
            temperature=0.7,
        )
        import re
        questions = []
        for line in raw.strip().splitlines():
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
            if cleaned and len(cleaned) > 10 and cleaned.endswith("?"):
                questions.append(cleaned)
        return questions[:n]

    # -- Stats --

    def stats(self) -> dict:
        store_stats = self.store.stats()
        return {
            "concepts": store_stats["concepts"],
            "associations": store_stats["associations"],
            "learn_count": self._learn_count,
            "query_count": self._query_count,
            "clip_ready": self.clip.ready,
            "llm_ready": self.llm.ready,
        }

    @staticmethod
    def _make_id(concept: str) -> str:
        return hashlib.sha256(concept.encode()).hexdigest()[:16]

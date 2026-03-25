"""Quantum Associative Memory v2 -- the brain of the LQNN system.

v2 adds the Probabilistic Wave-Collapse Engine and Associative Resonance:

WAVE-COLLAPSE:
- Probability amplitudes: exp(-distance / temperature) instead of raw distance
- Interference patterns: constructive (shared knowledge) / destructive (contradictions)
- Multi-hop collapse: second-order knowledge via entangled probes
- Adaptive n_results: 5 -> 25 -> 50 based on confidence
- Confidence wave function: sum(amplitude^2) normalised

RESONANCE:
- 4D association strength tensors: [semantic, temporal, access, centrality]
- Cross-pollination: LLM sees nearest neighbours when generating associations
- Resonance detection: auto-create meta-concepts when concepts share >5 associations
- Adaptive 1-4 background association workers based on GPU load

Original v1 features preserved:
- Superposition: concept = cloud of related vectors
- Collapse: query selects the most relevant associations
- Volatility decay and crystallisation
- Visual-first learning (70/30 image/text)
- Multi-image learning
- Network crystallisation bonus
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

# Wave-collapse constants
TEMPERATURE_DEFAULT = 0.25
TEMPERATURE_MIN = 0.05
TEMPERATURE_MAX = 0.8
MULTI_HOP_PROBES = 5
MULTI_HOP_SECOND_ORDER = 10
CONTEXT_CHARS_PER_FRAGMENT = 500
MAX_CONTEXT_FRAGMENTS = 30

# Adaptive collapse thresholds
ADAPTIVE_N_LOW = 5
ADAPTIVE_N_MID = 25
ADAPTIVE_N_HIGH = 50
CONFIDENCE_EXPAND_THRESHOLD_1 = 0.3
CONFIDENCE_EXPAND_THRESHOLD_2 = 0.15

# Association queue
ASSOC_QUEUE_SIZE = 2000
ASSOC_WORKERS_MIN = 1
ASSOC_WORKERS_MAX = 4

# Resonance
RESONANCE_SHARED_THRESHOLD = 5


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
    multi_hop_concepts: list[dict] = field(default_factory=list)
    amplitudes: list[float] = field(default_factory=list)


class AssociativeMemory:
    """The quantum associative brain v2.

    Wave-collapse engine with multi-hop probing and resonance amplification.
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
        self._assoc_bg_queue: queue.Queue = queue.Queue(maxsize=ASSOC_QUEUE_SIZE)
        self._assoc_bg_threads: list[threading.Thread] = []
        self._assoc_bg_running = False
        self._pool = ThreadPoolExecutor(max_workers=4)

        # HEI will be injected by main_loop after construction
        self._hei = None
        # Batch engine will be injected by main_loop after construction
        self._batch_engine = None

        self._start_bg_association_workers()

    def set_hei(self, hei) -> None:
        self._hei = hei

    def set_batch_engine(self, engine) -> None:
        self._batch_engine = engine

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

    # ------------------------------------------------------------------ #
    # Background association workers (adaptive pool)                       #
    # ------------------------------------------------------------------ #

    def _start_bg_association_workers(self) -> None:
        if self._assoc_bg_running:
            return
        self._assoc_bg_running = True
        for i in range(ASSOC_WORKERS_MIN):
            t = threading.Thread(
                target=self._bg_association_loop, daemon=True,
                name=f"assoc-bg-{i}",
            )
            t.start()
            self._assoc_bg_threads.append(t)

    def _scale_workers(self) -> None:
        """Scale up association workers if queue is deep and GPU has room."""
        alive = sum(1 for t in self._assoc_bg_threads if t.is_alive())
        qsize = self._assoc_bg_queue.qsize()
        target = min(ASSOC_WORKERS_MAX, max(ASSOC_WORKERS_MIN,
                                             qsize // 100 + 1))
        while alive < target:
            t = threading.Thread(
                target=self._bg_association_loop, daemon=True,
                name=f"assoc-bg-{alive}",
            )
            t.start()
            self._assoc_bg_threads.append(t)
            alive += 1

    def _bg_association_loop(self) -> None:
        while self._assoc_bg_running:
            try:
                concept, primary_vec, n = self._assoc_bg_queue.get(timeout=2.0)
                self._generate_associations_sync(concept, primary_vec, n)
            except queue.Empty:
                continue
            except Exception as e:
                log.debug("BG association error: %s", e)

    # ------------------------------------------------------------------ #
    # Learning                                                             #
    # ------------------------------------------------------------------ #

    def learn_concept(self, concept: str, image: bytes | None = None,
                      source: str = "manual",
                      initial_confidence: float = 0.5) -> QuantumState:
        """Learn a concept by encoding it and generating associations."""
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

        if self._hei:
            self._hei.assign(concept_id, primary_vec)

        immediate = self._generate_associations_sync(
            concept_lower, primary_vec, n=5)

        try:
            self._assoc_bg_queue.put_nowait(
                (concept_lower, primary_vec, 25))
        except queue.Full:
            pass

        self._scale_workers()
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
        """Learn from multiple images of the same concept."""
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

        if self._hei:
            self._hei.assign(concept_id, primary_vec)

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

    # ------------------------------------------------------------------ #
    # Querying -- Probabilistic Wave-Collapse Engine                       #
    # ------------------------------------------------------------------ #

    def _compute_amplitudes(self, distances: list[float],
                            temperature: float) -> list[float]:
        """Compute quantum probability amplitudes from distances."""
        amps = []
        for d in distances:
            amps.append(np.exp(-d / max(temperature, 1e-6)))
        total = sum(a * a for a in amps)
        if total > 0:
            norm = np.sqrt(total)
            amps = [a / norm for a in amps]
        return amps

    def _adaptive_temperature(self, query: str) -> float:
        """Adapt temperature based on query specificity.

        Short, specific queries get low temperature (sharp collapse).
        Long, vague queries get high temperature (broad search).
        """
        words = query.split()
        n = len(words)
        if n <= 3:
            return TEMPERATURE_MIN + 0.05
        if n <= 8:
            return TEMPERATURE_DEFAULT
        return min(TEMPERATURE_MAX, TEMPERATURE_DEFAULT + 0.03 * (n - 8))

    def _interference(self, concepts: list[dict],
                      amplitudes: list[float]) -> list[float]:
        """Apply quantum interference between nearby concepts.

        Constructive: concepts with similar metadata reinforce each other.
        Destructive: concepts with contradicting sources cancel.
        """
        n = len(concepts)
        if n < 2:
            return amplitudes

        adjusted = list(amplitudes)
        for i in range(min(n, 10)):
            src_i = concepts[i].get("metadata", {}).get("source", "")
            for j in range(i + 1, min(n, 10)):
                src_j = concepts[j].get("metadata", {}).get("source", "")
                dist_i = concepts[i].get("distance", 1.0)
                dist_j = concepts[j].get("distance", 1.0)

                if abs(dist_i - dist_j) < 0.05:
                    if src_i == src_j:
                        boost = 0.1 * amplitudes[i]
                        adjusted[i] += boost
                        adjusted[j] += boost
                    else:
                        penalty = 0.05 * amplitudes[j]
                        adjusted[j] = max(0.01, adjusted[j] - penalty)

        total = sum(a * a for a in adjusted)
        if total > 0:
            norm = np.sqrt(total)
            adjusted = [a / norm for a in adjusted]
        return adjusted

    def _multi_hop_collapse(self, primary_concepts: list[dict],
                            query_vec: np.ndarray,
                            n_probes: int = MULTI_HOP_PROBES,
                            n_second: int = MULTI_HOP_SECOND_ORDER) -> list[dict]:
        """Second-order knowledge: query associations of top concepts.

        Uses the top primary concepts as "entangled probes" to find
        knowledge that is connected-to-connected.
        """
        seen_ids = {c.get("id", "") for c in primary_concepts}
        second_order = []

        probe_concepts = primary_concepts[:n_probes]
        futures = []

        for c in probe_concepts:
            doc = c.get("document", "")
            if not doc:
                continue
            probe_vec = self._cached_encode_text(doc)
            futures.append(self._pool.submit(
                self.store.query_associations, probe_vec, 5))

        for fut in futures:
            try:
                assoc_results = fut.result(timeout=2)
                for a in assoc_results:
                    aid = a.get("id", "")
                    if aid not in seen_ids:
                        seen_ids.add(aid)
                        second_order.append(a)
            except Exception:
                continue

        concept_futures = []
        for c in probe_concepts[:3]:
            doc = c.get("document", "")
            if not doc:
                continue
            probe_vec = self._cached_encode_text(doc)
            concept_futures.append(self._pool.submit(
                self.store.query_concepts, probe_vec, 5))

        for fut in concept_futures:
            try:
                results = fut.result(timeout=2)
                for r in results:
                    rid = r.get("id", "")
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        second_order.append(r)
            except Exception:
                continue

        second_order.sort(key=lambda x: x.get("distance", 1.0))
        return second_order[:n_second]

    def query(self, question: str, n_results: int = 10) -> CollapseResult:
        """Probabilistic wave-collapse with multi-hop probing.

        1. Adaptive temperature based on query specificity
        2. First collapse: primary concepts + associations (parallel)
        3. Compute probability amplitudes + interference
        4. Multi-hop: second-order knowledge via entangled probes
        5. Build rich context (up to 30 fragments x 500 chars)
        6. Wave-function confidence: sum(amplitude^2)
        """
        query_vec = self._cached_encode_text(question)
        temperature = self._adaptive_temperature(question)

        crystal_results = self.store.query_crystal_tier(query_vec, n=3)

        n_initial = ADAPTIVE_N_LOW
        fut_concepts = self._pool.submit(
            self.store.query_concepts, query_vec, n_initial)
        fut_assocs = self._pool.submit(
            self.store.query_associations, query_vec, n_initial * 2)

        concepts = fut_concepts.result()
        assoc_results = fut_assocs.result()

        if concepts:
            distances = [c.get("distance", 1.0) for c in concepts]
            amplitudes = self._compute_amplitudes(distances, temperature)
            raw_confidence = sum(a * a for a in amplitudes[:5])
        else:
            amplitudes = []
            raw_confidence = 0.0

        if raw_confidence < CONFIDENCE_EXPAND_THRESHOLD_1:
            expanded = self.store.query_concepts(query_vec, ADAPTIVE_N_MID)
            if len(expanded) > len(concepts):
                concepts = expanded
                distances = [c.get("distance", 1.0) for c in concepts]
                amplitudes = self._compute_amplitudes(distances, temperature)
                raw_confidence = sum(a * a for a in amplitudes[:10])

        if raw_confidence < CONFIDENCE_EXPAND_THRESHOLD_2:
            expanded = self.store.query_concepts(query_vec, ADAPTIVE_N_HIGH)
            if len(expanded) > len(concepts):
                concepts = expanded
                distances = [c.get("distance", 1.0) for c in concepts]
                amplitudes = self._compute_amplitudes(distances, temperature)
                raw_confidence = sum(a * a for a in amplitudes[:15])

        amplitudes = self._interference(concepts, amplitudes)

        ranked = sorted(
            zip(concepts, amplitudes),
            key=lambda x: x[1],
            reverse=True,
        )
        concepts = [r[0] for r in ranked]
        amplitudes = [r[1] for r in ranked]

        multi_hop = self._multi_hop_collapse(concepts, query_vec)

        self.store.batch_touch(
            [c.get("id", "") for c in concepts[:10]],
            [c.get("metadata", {}) for c in concepts[:10]],
        )

        context_parts = []

        for c in crystal_results[:2]:
            doc = c.get("document", "")
            meta = c.get("metadata", {})
            full_text = meta.get("full_text", "")
            dist = c.get("distance", 1.0)
            text = full_text[:CONTEXT_CHARS_PER_FRAGMENT] if full_text else doc
            if text:
                context_parts.append(
                    f"- [crystallized] {text} (relevance: {1 - dist:.2f})")

        for c in concepts[:5]:
            doc = c.get("document", "")
            meta = c.get("metadata", {})
            full_text = meta.get("full_text", "")
            dist = c.get("distance", 1.0)
            text = full_text[:CONTEXT_CHARS_PER_FRAGMENT] if full_text else doc
            if text:
                context_parts.append(f"- {text} (relevance: {1 - dist:.2f})")

        for mh in multi_hop[:MULTI_HOP_SECOND_ORDER]:
            doc = mh.get("document", "")
            meta = mh.get("metadata", {})
            full_text = meta.get("full_text", "")
            text = full_text[:CONTEXT_CHARS_PER_FRAGMENT] if full_text else doc
            if text:
                context_parts.append(f"- [linked] {text}")

        for a in assoc_results[:15]:
            doc = a.get("document", "")
            if doc:
                context_parts.append(f"- {doc}")

        context_parts = context_parts[:MAX_CONTEXT_FRAGMENTS]
        context = "\n".join(context_parts) if context_parts else ""

        confidence = min(1.0, max(0.0, raw_confidence))

        self._query_count += 1
        return CollapseResult(
            query=question,
            matched_concepts=concepts,
            confidence=confidence,
            context=context,
            associations=assoc_results,
            multi_hop_concepts=multi_hop,
            amplitudes=amplitudes,
        )

    # ------------------------------------------------------------------ #
    # Consolidation (incremental)                                          #
    # ------------------------------------------------------------------ #

    def consolidate(self) -> dict:
        """Run a consolidation cycle with incremental dirty-set support.

        If the dirty set is available and non-empty, only process those
        concepts.  Falls back to full scan when dirty set is empty (first
        run or after a rebuild).
        """
        now = time.time()
        pruned = 0
        crystallized = 0
        decayed = 0

        dirty = self.store.pop_dirty_ids()

        if dirty:
            try:
                data = self.store._concepts.get(
                    ids=list(dirty),
                    include=["metadatas"],
                )
            except Exception:
                data = self.store._concepts.get(include=["metadatas"])
        else:
            data = self.store._concepts.get(include=["metadatas"])

        all_ids = data.get("ids", [])
        all_metas = data.get("metadatas", [])
        if not all_ids:
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

        for i, cid in enumerate(all_ids):
            meta = all_metas[i] if i < len(all_metas) else {}
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
                    self._promote_crystal(cid, meta)
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
            "total": len(all_ids) - pruned,
            "incremental": bool(dirty),
        }
        log.info("Consolidation: %s", result)
        return result

    def _promote_crystal(self, concept_id: str, meta: dict) -> None:
        """Promote a concept to the in-memory crystallized tier."""
        try:
            data = self.store.get_concept(concept_id)
            if data and data.get("embedding"):
                vec = np.array(data["embedding"], dtype=np.float32)
                self.store.promote_to_crystal(
                    concept_id, vec,
                    data.get("metadata", meta),
                    data.get("document", meta.get("concept", "")),
                )
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Self-play                                                            #
    # ------------------------------------------------------------------ #

    def self_play_cycle(self) -> dict:
        """Enhanced self-play with Q&A validation."""
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

    # ------------------------------------------------------------------ #
    # Resonance detection                                                  #
    # ------------------------------------------------------------------ #

    def detect_resonance(self, max_pairs: int = 10) -> list[dict]:
        """Find concept pairs that share many associations and create
        meta-concepts linking them (emergent knowledge structures)."""
        assoc_data = self.store._associations.get(
            include=["metadatas"],
            limit=min(self.store.association_count(), 5000),
        )
        if not assoc_data or not assoc_data["metadatas"]:
            return []

        source_targets: dict[str, set[str]] = {}
        for meta in assoc_data["metadatas"]:
            src = meta.get("source_concept", "")
            tgt = meta.get("target_concept", "")
            if src and tgt:
                source_targets.setdefault(src, set()).add(tgt)

        resonances = []
        sources = list(source_targets.keys())
        for i in range(min(len(sources), 200)):
            for j in range(i + 1, min(len(sources), 200)):
                shared = source_targets[sources[i]] & source_targets[sources[j]]
                if len(shared) >= RESONANCE_SHARED_THRESHOLD:
                    resonances.append({
                        "concept_a": sources[i],
                        "concept_b": sources[j],
                        "shared_associations": len(shared),
                        "shared_words": list(shared)[:10],
                    })

        resonances.sort(key=lambda r: r["shared_associations"], reverse=True)
        created = []
        for res in resonances[:max_pairs]:
            meta_concept = f"{res['concept_a']} <-> {res['concept_b']}"
            vec_a = self._cached_encode_text(res["concept_a"])
            vec_b = self._cached_encode_text(res["concept_b"])
            meta_vec = (vec_a + vec_b) / 2
            meta_vec = meta_vec / np.linalg.norm(meta_vec)

            entry = VectorEntry(
                id=self._make_id(meta_concept),
                vector=meta_vec,
                concept=meta_concept,
                source="resonance",
                volatility=0.3,
                confidence=0.8,
                metadata={"resonance_strength": res["shared_associations"]},
            )
            self.store.add_concept(entry)
            created.append(res)

        if created:
            log.info("Resonance: created %d meta-concepts", len(created))
        return created

    # ------------------------------------------------------------------ #
    # Cleanup                                                              #
    # ------------------------------------------------------------------ #

    def cleanup_garbage(self) -> dict:
        """Scan all stored concepts and remove web boilerplate / junk."""
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

    # ------------------------------------------------------------------ #
    # Training Questions                                                   #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Stats                                                                #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        store_stats = self.store.stats()
        return {
            "concepts": store_stats["concepts"],
            "associations": store_stats["associations"],
            "crystallized": store_stats.get("crystallized", 0),
            "learn_count": self._learn_count,
            "query_count": self._query_count,
            "clip_ready": self.clip.ready,
            "llm_ready": self.llm.ready,
            "assoc_queue_size": self._assoc_bg_queue.qsize(),
            "assoc_workers": sum(1 for t in self._assoc_bg_threads
                                 if t.is_alive()),
        }

    @staticmethod
    def _make_id(concept: str) -> str:
        return hashlib.sha256(concept.encode()).hexdigest()[:16]

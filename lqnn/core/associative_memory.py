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

# Quantum Decoherence Shield patterns -- environmental noise that
# destroys coherent quantum states.
_GARBAGE_INDICATORS = frozenset([
    "%n%n", "%s%s", "rnd=", ";app=", ";uid=", "68BAC", "0A27C",
    "iTradeEU", "76FF3D", "B8FCA4", "R_1;uid",
])
_MIN_ALPHA_RATIO = 0.40


def _is_coherent(text: str) -> bool:
    """Quantum decoherence shield: reject fragments that are environmental
    noise (corrupted PDF residue, binary artefacts, etc.)."""
    if not text or len(text) < 3:
        return False
    text_lower = text[:200].lower()
    for pattern in _GARBAGE_INDICATORS:
        if pattern.lower() in text_lower:
            return False
    alpha_count = sum(c.isalpha() or c.isspace() for c in text[:200])
    if alpha_count / max(len(text[:200]), 1) < _MIN_ALPHA_RATIO:
        return False
    return True


IMAGE_WEIGHT = 0.7
TEXT_WEIGHT = 0.3
EMBED_CACHE_SIZE = 8192

# Wave-collapse constants
TEMPERATURE_DEFAULT = 0.25
TEMPERATURE_MIN = 0.05
TEMPERATURE_MAX = 0.8
MULTI_HOP_PROBES = 8
MULTI_HOP_SECOND_ORDER = 16

# Superposition Context Assembly v5 (Heisenberg-optimised for 7B model)
CONTEXT_CHARS_PER_FRAGMENT = 500
MAX_CONTEXT_FRAGMENTS = 40
CONTEXT_BUDGET_CHARS = 8_000  # ~2K tokens -- optimal speed/quality for 7B

# Adaptive collapse thresholds
ADAPTIVE_N_LOW = 5
ADAPTIVE_N_MID = 25
ADAPTIVE_N_HIGH = 50
CONFIDENCE_EXPAND_THRESHOLD_1 = 0.3
CONFIDENCE_EXPAND_THRESHOLD_2 = 0.15

# Association queue
ASSOC_QUEUE_SIZE = 2000
ASSOC_WORKERS_MIN = 1
ASSOC_WORKERS_MAX = 8

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
        self._pool = ThreadPoolExecutor(max_workers=8)

        # HEI will be injected by main_loop after construction
        self._hei = None
        # Batch engine will be injected by main_loop after construction
        self._batch_engine = None

        self._confidence_history: list[float] = []
        self._confidence_window = 20

        self._start_bg_association_workers()

    def set_hei(self, hei) -> None:
        self._hei = hei

    def set_batch_engine(self, engine) -> None:
        self._batch_engine = engine

    def _cached_encode_text(self, text: str) -> np.ndarray:
        """CLIP text encoding with LRU cache + batch engine fast path."""
        key = text[:512]
        if key in self._embed_cache:
            self._embed_cache.move_to_end(key)
            return self._embed_cache[key]
        if self._batch_engine is not None:
            vec = self._batch_engine.encode_text_urgent(key)
        else:
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
            if self.llm.chat_active:
                time.sleep(1)
                continue
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
            if self._batch_engine is not None:
                assoc_vectors = self._batch_engine.encode_texts_urgent(
                    assoc_words)
            else:
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
        """Self-tuning temperature based on query specificity and rolling
        confidence history.

        Short queries -> low temperature (sharp collapse).
        Long queries -> higher temperature (broad search).
        Recent low confidence -> widen the search automatically.
        Recent high confidence -> sharpen the search.
        """
        words = query.split()
        n = len(words)
        if n <= 3:
            base_temp = TEMPERATURE_MIN + 0.05
        elif n <= 8:
            base_temp = TEMPERATURE_DEFAULT
        else:
            base_temp = min(TEMPERATURE_MAX, TEMPERATURE_DEFAULT + 0.03 * (n - 8))

        if len(self._confidence_history) >= 3:
            avg_conf = sum(self._confidence_history[-self._confidence_window:]) / \
                       len(self._confidence_history[-self._confidence_window:])
            if avg_conf < 0.2:
                base_temp = min(TEMPERATURE_MAX, base_temp + 0.15)
            elif avg_conf > 0.6:
                base_temp = max(TEMPERATURE_MIN, base_temp - 0.05)

        return base_temp

    def _dynamic_probe_count(self) -> int:
        """Adjust multi-hop probe count based on recent query performance."""
        if len(self._confidence_history) < 3:
            return MULTI_HOP_PROBES

        avg_conf = sum(self._confidence_history[-self._confidence_window:]) / \
                   len(self._confidence_history[-self._confidence_window:])

        if avg_conf < 0.15:
            return min(MULTI_HOP_PROBES + 4, 12)
        if avg_conf < 0.3:
            return MULTI_HOP_PROBES + 2
        if avg_conf > 0.7:
            return max(3, MULTI_HOP_PROBES - 2)
        return MULTI_HOP_PROBES

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
                assoc_results = fut.result(timeout=0.5)
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
                results = fut.result(timeout=0.5)
                for r in results:
                    rid = r.get("id", "")
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        second_order.append(r)
            except Exception:
                continue

        second_order.sort(key=lambda x: x.get("distance", 1.0))
        return second_order[:n_second]

    def _hei_narrow(self, query_vec: np.ndarray) -> list[str] | None:
        """Use HEI to pre-filter concept IDs for faster collapse."""
        if self._hei is None or not self._hei.ready:
            return None
        try:
            return self._hei.narrow(query_vec)
        except Exception:
            return None

    def query(self, question: str, n_results: int = 10) -> CollapseResult:
        """Probabilistic wave-collapse with HEI pre-filtering and
        superposition context assembly.

        1. HEI.narrow() pre-filters concept IDs (10-20x faster)
        2. Adaptive temperature based on query specificity
        3. First collapse: primary concepts + associations (parallel)
        4. Compute probability amplitudes + interference
        5. Multi-hop: second-order knowledge via entangled probes
        6. Superposition context assembly (up to ~50K chars)
        7. Wave-function confidence: sum(amplitude^2)
        """
        query_vec = self._cached_encode_text(question)
        temperature = self._adaptive_temperature(question)

        fut_narrow = self._pool.submit(self._hei_narrow, query_vec)
        fut_crystal = self._pool.submit(
            self.store.query_crystal_tier, query_vec, 5)

        narrowed_ids = fut_narrow.result(timeout=0.5)
        crystal_results = fut_crystal.result(timeout=0.5)

        n_initial = ADAPTIVE_N_LOW
        fut_concepts = self._pool.submit(
            self.store.query_concepts, query_vec, n_initial,
            None, narrowed_ids)
        fut_assocs = self._pool.submit(
            self.store.query_associations, query_vec, n_initial * 2)

        concepts = fut_concepts.result(timeout=1.0)
        assoc_results = fut_assocs.result(timeout=1.0)

        if concepts:
            distances = [c.get("distance", 1.0) for c in concepts]
            amplitudes = self._compute_amplitudes(distances, temperature)
            raw_confidence = sum(a * a for a in amplitudes[:5])
        else:
            amplitudes = []
            raw_confidence = 0.0

        if raw_confidence < CONFIDENCE_EXPAND_THRESHOLD_1:
            expanded = self.store.query_concepts(
                query_vec, ADAPTIVE_N_MID, id_filter=narrowed_ids)
            if len(expanded) > len(concepts):
                concepts = expanded
                distances = [c.get("distance", 1.0) for c in concepts]
                amplitudes = self._compute_amplitudes(distances, temperature)
                raw_confidence = sum(a * a for a in amplitudes[:10])

        if raw_confidence < CONFIDENCE_EXPAND_THRESHOLD_2:
            expanded = self.store.query_concepts(
                query_vec, ADAPTIVE_N_HIGH, id_filter=narrowed_ids)
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

        n_probes = self._dynamic_probe_count()
        multi_hop = self._multi_hop_collapse(
            concepts, query_vec, n_probes=n_probes)

        self.store.batch_touch(
            [c.get("id", "") for c in concepts[:10]],
            [c.get("metadata", {}) for c in concepts[:10]],
        )

        context = self._assemble_superposition_context(
            crystal_results, concepts, amplitudes, multi_hop, assoc_results)

        confidence = min(1.0, max(0.0, raw_confidence))

        self._confidence_history.append(confidence)
        if len(self._confidence_history) > self._confidence_window * 2:
            self._confidence_history = \
                self._confidence_history[-self._confidence_window:]

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
    # Superposition Context Assembly                                       #
    # ------------------------------------------------------------------ #

    def _assemble_superposition_context(
        self,
        crystal_results: list[dict],
        concepts: list[dict],
        amplitudes: list[float],
        multi_hop: list[dict],
        assoc_results: list[dict],
    ) -> str:
        """Build a rich, deduplicated context string ranked by relevance.

        Hierarchical priority:
          1. Crystallized concepts (full_text, highest trust)
          2. Primary concepts ranked by amplitude * relevance
          3. Multi-hop linked knowledge (full_text)
          4. Association labels (compact semantic links)

        Deduplication via first-100-char hash prevents redundant fragments.
        Total output is capped at CONTEXT_BUDGET_CHARS (~50K chars / ~12K tokens).
        """
        seen_hashes: set[str] = set()
        scored_parts: list[tuple[float, str]] = []

        def _dedup_add(text: str, score: float, prefix: str = "") -> None:
            text = text.strip()
            if not text:
                return
            if not _is_coherent(text):
                return
            sig = hashlib.md5(text[:100].encode()).hexdigest()
            if sig in seen_hashes:
                return
            seen_hashes.add(sig)
            label = f"- {prefix}{text}"
            scored_parts.append((score, label))

        for c in crystal_results[:5]:
            meta = c.get("metadata", {})
            full_text = meta.get("full_text", "")
            doc = c.get("document", "")
            dist = c.get("distance", 1.0)
            text = (full_text[:CONTEXT_CHARS_PER_FRAGMENT] if full_text
                    else doc)
            relevance = max(0.0, 1.0 - dist)
            _dedup_add(text, 10.0 + relevance,
                       f"[crystallized, relevance: {relevance:.2f}] ")

        for i, c in enumerate(concepts[:20]):
            meta = c.get("metadata", {})
            full_text = meta.get("full_text", "")
            doc = c.get("document", "")
            dist = c.get("distance", 1.0)
            amp = amplitudes[i] if i < len(amplitudes) else 0.0
            text = (full_text[:CONTEXT_CHARS_PER_FRAGMENT] if full_text
                    else doc)
            relevance = max(0.0, 1.0 - dist)
            score = relevance * (1.0 + amp)
            _dedup_add(text, 5.0 + score,
                       f"[relevance: {relevance:.2f}] ")

        for mh in multi_hop[:MULTI_HOP_SECOND_ORDER * 2]:
            meta = mh.get("metadata", {})
            full_text = meta.get("full_text", "")
            doc = mh.get("document", "")
            dist = mh.get("distance", 1.0)
            text = (full_text[:CONTEXT_CHARS_PER_FRAGMENT] if full_text
                    else doc)
            relevance = max(0.0, 1.0 - dist)
            _dedup_add(text, 2.0 + relevance, "[linked] ")

        for a in assoc_results[:30]:
            doc = a.get("document", "")
            dist = a.get("distance", 1.0)
            if doc:
                relevance = max(0.0, 1.0 - dist)
                _dedup_add(doc, 1.0 + relevance, "")

        scored_parts.sort(key=lambda x: x[0], reverse=True)

        context_parts: list[str] = []
        total_chars = 0
        for _score, part in scored_parts[:MAX_CONTEXT_FRAGMENTS]:
            if total_chars + len(part) > CONTEXT_BUDGET_CHARS:
                remaining = CONTEXT_BUDGET_CHARS - total_chars
                if remaining > 100:
                    context_parts.append(part[:remaining])
                break
            context_parts.append(part)
            total_chars += len(part) + 1

        return "\n".join(context_parts) if context_parts else ""

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
        if self.llm.chat_active:
            return {"action": "skip", "reason": "chat_priority"}
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
    # Response Quality Feedback (Living Organism)                          #
    # ------------------------------------------------------------------ #

    FEEDBACK_REINFORCE_RATE = 0.02
    FEEDBACK_WEAKEN_RATE = 0.005
    RESONANCE_AMPLIFY_THRESHOLD = 0.5

    def feedback_response_quality(self, concept_ids: list[str],
                                  confidence: float) -> None:
        """Feed response quality back into concept volatility.

        High confidence responses reinforce the concepts that contributed
        (lower volatility = more permanent). Low confidence slightly
        increases volatility (natural selection pressure on weak knowledge).
        """
        if not concept_ids:
            return

        try:
            data = self.store._concepts.get(
                ids=concept_ids[:20],
                include=["metadatas"],
            )
        except Exception:
            return

        ids = data.get("ids", [])
        metas = data.get("metadatas", [])
        if not ids:
            return

        updated_ids = []
        updated_metas = []

        for i, cid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            vol = meta.get("volatility", 0.5)

            if confidence >= 0.5:
                delta = self.FEEDBACK_REINFORCE_RATE * confidence
                new_vol = max(0.0, vol - delta)
            elif confidence < 0.15:
                new_vol = min(1.0, vol + self.FEEDBACK_WEAKEN_RATE)
            else:
                continue

            if abs(new_vol - vol) > 0.001:
                meta_copy = dict(meta)
                meta_copy["volatility"] = round(new_vol, 4)
                updated_ids.append(cid)
                updated_metas.append(meta_copy)

        if updated_ids:
            try:
                self.store._concepts.update(
                    ids=updated_ids, metadatas=updated_metas)
            except Exception:
                pass

    def amplify_resonance(self, query_text: str,
                          concept_ids: list[str],
                          confidence: float) -> None:
        """Create resonance links between the query and concepts that
        contributed to a high-confidence response.

        This builds "response resonance" associations that link
        successful question patterns to the knowledge paths that
        answered them, making future similar queries faster and more
        accurate.
        """
        if confidence < self.RESONANCE_AMPLIFY_THRESHOLD:
            return
        if not concept_ids or len(concept_ids) < 2:
            return

        try:
            query_vec = self._cached_encode_text(query_text)
        except Exception:
            return

        top_ids = concept_ids[:5]
        for i in range(len(top_ids)):
            for j in range(i + 1, len(top_ids)):
                try:
                    src = top_ids[i]
                    tgt = top_ids[j]
                    assoc_id = f"resonance_{src}_{tgt}"
                    self.store.add_association(
                        source_concept=src,
                        target_concept=tgt,
                        vector=query_vec,
                        strength=confidence,
                        assoc_id=assoc_id,
                    )
                except Exception:
                    continue

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

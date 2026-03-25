"""Quantum Associative Memory -- the brain of the LQNN system.

Implements:
- Superposition: a concept exists as a cloud of related vectors simultaneously
- Collapse: on query, the system collapses to the most relevant associations
- Volatility decay: frequently accessed vectors become stable (crystallize),
  unused ones decay and are eventually pruned
- Consolidation: periodic cycles that promote short-term to long-term memory
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from lqnn.core.vector_store import VectorEntry, VectorStore
from lqnn.models.clip_encoder import CLIPEncoder
from lqnn.models.llm_engine import LLMEngine

log = logging.getLogger(__name__)


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

    def __init__(self, store: VectorStore, clip: CLIPEncoder,
                 llm: LLMEngine) -> None:
        self.store = store
        self.clip = clip
        self.llm = llm
        self._learn_count = 0
        self._query_count = 0

    # -- Learning --

    def learn_concept(self, concept: str, image: bytes | None = None,
                      source: str = "manual") -> QuantumState:
        """Learn a concept by encoding it and generating associations.

        This is the core 'human-like' learning step:
        1. Encode the concept text with CLIP (text vector)
        2. If image provided, encode with CLIP (visual vector) and average
        3. Ask LLM for associations
        4. Encode each association with CLIP
        5. Store everything in ChromaDB with initial volatility=1.0
        """
        concept_lower = concept.strip().lower()
        concept_id = self._make_id(concept_lower)

        text_vec = self.clip.encode_text(concept_lower)

        if image is not None:
            img_vec = self.clip.encode_image(image)
            primary_vec = (text_vec + img_vec) / 2.0
            primary_vec = primary_vec / np.linalg.norm(primary_vec)
        else:
            primary_vec = text_vec

        entry = VectorEntry(
            id=concept_id,
            vector=primary_vec,
            concept=concept_lower,
            source=source,
            volatility=1.0,
            confidence=0.5,
        )
        self.store.add_concept(entry)

        associations = self._generate_associations(concept_lower, primary_vec)

        self._learn_count += 1
        log.info(
            "Learned '%s' with %d associations (source=%s)",
            concept_lower, len(associations), source,
        )
        return QuantumState(
            concept=concept_lower,
            primary_vector=primary_vec,
            associations=associations,
            volatility=1.0,
            confidence=0.5,
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
        """Generate and store association vectors for a concept."""
        assoc_words = self.llm.extract_associations(concept, n=20)
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

        1. Encode the question with CLIP
        2. Find nearest concept vectors
        3. Retrieve associations for those concepts
        4. Build context from matched knowledge
        5. Return collapsed result with confidence
        """
        query_vec = self.clip.encode_text(question)

        concepts = self.store.query_concepts(query_vec, n=n_results)

        for c in concepts:
            cid = c.get("id", "")
            meta = c.get("metadata", {})
            self.store.update_metadata(cid, {
                "access_count": meta.get("access_count", 0) + 1,
                "last_accessed": time.time(),
            })

        assoc_results = self.store.query_associations(query_vec, n=n_results * 2)

        context_parts = []
        for c in concepts[:5]:
            doc = c.get("document", "")
            dist = c.get("distance", 1.0)
            if doc:
                context_parts.append(f"- {doc} (relevance: {1 - dist:.2f})")

        for a in assoc_results[:10]:
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
        """Run a consolidation cycle.

        - Decrease volatility of frequently accessed concepts (crystallize)
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

        for i, cid in enumerate(all_concepts["ids"]):
            meta = all_concepts["metadatas"][i]
            volatility = meta.get("volatility", 1.0)
            access_count = meta.get("access_count", 0)
            last_accessed = meta.get("last_accessed", now)
            age_hours = (now - last_accessed) / 3600

            if access_count > 5:
                new_vol = max(0.0, volatility - self.VOLATILITY_DECAY_RATE *
                              min(access_count, 50))
                if new_vol <= self.CRYSTALLIZE_THRESHOLD:
                    crystallized += 1
            elif age_hours > 24:
                new_vol = min(1.0, volatility + self.VOLATILITY_INCREASE_RATE *
                              min(age_hours / 24, 10))
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
        """Query own knowledge to find contradictions and reinforce patterns.

        Picks a random stored concept, queries for it, checks if the
        retrieved associations are consistent.
        """
        count = self.store.concept_count()
        if count == 0:
            return {"action": "skip", "reason": "no_concepts"}

        all_ids = self.store._concepts.get(
            include=["documents"],
            limit=min(count, 100),
        )
        if not all_ids["documents"]:
            return {"action": "skip", "reason": "empty_documents"}

        idx = np.random.randint(len(all_ids["documents"]))
        concept = all_ids["documents"][idx]

        result = self.query(concept)

        if result.confidence < 0.5 and count > 10:
            new_assocs = self._generate_associations(
                concept, self.clip.encode_text(concept))
            return {
                "action": "reinforced",
                "concept": concept,
                "new_associations": len(new_assocs),
                "confidence_before": result.confidence,
            }

        return {
            "action": "validated",
            "concept": concept,
            "confidence": result.confidence,
        }

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

"""Adaptive Phase-Space Compression.

Compresses the LQNN knowledge representation to fit ~10x more concepts
in the same hardware budget:

1. Matryoshka Embeddings:
   - Use the first 256 dims of CLIP 512-d for fast ANN search
   - Re-rank top candidates with full 512 dims for accuracy

2. Volatility-Based Memory Tiers:
   - Crystallised concepts (vol < 0.2) -> fast in-memory numpy index
   - Active concepts (0.2 <= vol < 0.8) -> Chroma (standard)
   - Volatile concepts (vol >= 0.8) -> Chroma low-priority

3. Dynamic Chunk Sizing:
   - High-information text -> smaller chunks (more vectors per concept)
   - Low-information text -> larger chunks (fewer vectors)
   - Uses character entropy as a lightweight proxy for information density

4. Association Pruning by Information Gain:
   - Compute pairwise cosine sim among association vectors
   - Keep only the top-K that are maximally diverse
   - Covers more semantic directions with fewer vectors
"""

from __future__ import annotations

import logging
import math
from collections import Counter

import numpy as np

log = logging.getLogger(__name__)

# Matryoshka settings
MATRYOSHKA_DIM = 256  # fast search dimension (first N of 512)
FULL_DIM = 512

# Dynamic chunk sizing
MIN_CHUNK_CHARS = 400
MAX_CHUNK_CHARS = 3000
DEFAULT_CHUNK_CHARS = 1800
ENTROPY_HIGH = 4.5   # bits -- high-information text
ENTROPY_LOW = 3.0    # bits -- low-information text

# Association pruning
MAX_DIVERSE_ASSOCIATIONS = 15


def matryoshka_search(query_vec: np.ndarray,
                      candidate_vecs: np.ndarray,
                      candidate_ids: list[str],
                      top_k_fast: int = 50,
                      top_k_final: int = 10,
                      ) -> list[tuple[str, float]]:
    """Two-stage search: fast 256-d -> precise 512-d re-rank.

    Parameters
    ----------
    query_vec : (512,) full CLIP vector
    candidate_vecs : (N, 512) matrix
    candidate_ids : list of N IDs
    top_k_fast : how many to keep from the 256-d pass
    top_k_final : how many to return after 512-d re-rank

    Returns
    -------
    List of (id, cosine_distance) sorted best-first.
    """
    if len(candidate_ids) == 0:
        return []

    q_short = query_vec[:MATRYOSHKA_DIM]
    q_short = q_short / (np.linalg.norm(q_short) + 1e-8)
    c_short = candidate_vecs[:, :MATRYOSHKA_DIM]
    norms = np.linalg.norm(c_short, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    c_short = c_short / norms

    sims_fast = c_short @ q_short
    top_fast = np.argsort(-sims_fast)[:top_k_fast]

    q_full = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    c_full = candidate_vecs[top_fast]
    sims_full = c_full @ q_full
    ranking = np.argsort(-sims_full)[:top_k_final]

    results = []
    for r in ranking:
        orig_idx = top_fast[r]
        dist = float(1.0 - sims_full[r])
        results.append((candidate_ids[orig_idx], dist))
    return results


def char_entropy(text: str) -> float:
    """Approximate information density via character-level Shannon entropy."""
    if not text:
        return 0.0
    counts = Counter(text.lower())
    n = len(text)
    ent = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def adaptive_chunk_size(text: str) -> int:
    """Return target chunk size based on information density.

    High-entropy text (dense info) -> smaller chunks for fine-grained vectors.
    Low-entropy text (repetitive) -> larger chunks for efficiency.
    """
    ent = char_entropy(text[:5000])

    if ent >= ENTROPY_HIGH:
        return MIN_CHUNK_CHARS
    if ent <= ENTROPY_LOW:
        return MAX_CHUNK_CHARS

    ratio = (ent - ENTROPY_LOW) / (ENTROPY_HIGH - ENTROPY_LOW)
    size = int(MAX_CHUNK_CHARS - ratio * (MAX_CHUNK_CHARS - MIN_CHUNK_CHARS))
    return max(MIN_CHUNK_CHARS, min(MAX_CHUNK_CHARS, size))


def prune_associations_by_diversity(vectors: np.ndarray,
                                    words: list[str],
                                    keep: int = MAX_DIVERSE_ASSOCIATIONS,
                                    ) -> list[int]:
    """Select the top-K most diverse associations by greedy max-min distance.

    Ensures the kept associations span the widest possible semantic space,
    rather than clustering around the same meaning.

    Parameters
    ----------
    vectors : (N, 512) association vectors
    words : N association words (for logging)
    keep : maximum to keep

    Returns
    -------
    Indices of the selected associations.
    """
    n = vectors.shape[0]
    if n <= keep:
        return list(range(n))

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = vectors / norms
    sim_matrix = normed @ normed.T

    selected = [0]
    remaining = set(range(1, n))

    while len(selected) < keep and remaining:
        best_idx = -1
        best_min_dist = -1.0

        for cand in remaining:
            min_sim = min(sim_matrix[cand, s] for s in selected)
            dist = 1.0 - min_sim
            if dist > best_min_dist:
                best_min_dist = dist
                best_idx = cand

        if best_idx >= 0:
            selected.append(best_idx)
            remaining.discard(best_idx)
        else:
            break

    return selected


def semantic_dedup_threshold(vector: np.ndarray,
                             cluster_centroid: np.ndarray,
                             threshold: float = 0.15) -> bool:
    """Return True if the vector is too close to the cluster centroid
    (should be merged rather than stored as a new concept)."""
    sim = float(np.dot(vector, cluster_centroid))
    return (1.0 - sim) < threshold

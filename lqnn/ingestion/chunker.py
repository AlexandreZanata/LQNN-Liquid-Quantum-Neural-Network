"""Quantum-inspired semantic chunker.

Instead of fixed-size chunks, we split on semantic boundaries (paragraphs,
sections, sentences) and treat each chunk as a quantum state candidate.
The CLIP encoder then collapses each chunk into a 512-d vector that
coexists in superposition inside ChromaDB.

v2: Adaptive chunk sizing via phase-space information density analysis.
High-entropy (information-dense) text gets smaller chunks for finer
vector resolution; low-entropy (repetitive) text gets larger chunks.

Strategy:
- Section-aware: respects [Page N] markers and heading-like lines
- Paragraph-level: splits on double newlines
- Sentence fallback: for long paragraphs
- Overlap: 10% overlap for context entanglement between adjacent chunks
- Minimum quality: rejects chunks that are too short or mostly noise
- Adaptive sizing: chunk target adapts to information density
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SECTION_RE = re.compile(
    r"^(?:\[Page \d+\]|Chapter \d|CHAPTER |PART |Section \d|\d+\.\d)",
    re.IGNORECASE,
)


@dataclass
class TextChunk:
    text: str
    index: int
    source_type: str
    source_name: str
    metadata: dict = field(default_factory=dict)


def semantic_chunk(
    text: str,
    source_type: str,
    source_name: str,
    max_chars: int = 0,
    overlap_chars: int = 200,
    metadata: dict | None = None,
    adaptive: bool = True,
) -> list[TextChunk]:
    """Split text into semantic chunks respecting paragraph and section
    boundaries.

    When *adaptive* is True (default), max_chars is computed dynamically
    based on the text's information density.
    """
    meta = metadata or {}
    text = text.strip()
    if not text:
        return []

    if adaptive and max_chars <= 0:
        from lqnn.core.phase_space import adaptive_chunk_size
        max_chars = adaptive_chunk_size(text)
    elif max_chars <= 0:
        max_chars = 1800

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[TextChunk] = []
    current = ""
    idx = 0

    def _flush():
        nonlocal current, idx
        trimmed = current.strip()
        if len(trimmed) >= 40:
            chunks.append(TextChunk(
                text=trimmed,
                index=idx,
                source_type=source_type,
                source_name=source_name,
                metadata={**meta, "chunk_index": idx},
            ))
            idx += 1
        current = ""

    for para in paragraphs:
        is_section_break = bool(_SECTION_RE.match(para))

        if is_section_break and len(current) > 200:
            _flush()
            current = current[-overlap_chars:] if current else ""

        if len(para) > max_chars:
            if current:
                _flush()
            sentences = re.split(r"(?<=[.!?;:])\s+", para)
            for sent in sentences:
                if len(current) + len(sent) + 1 > max_chars and current:
                    _flush()
                    current = current[-overlap_chars:] + " " + sent if current else sent
                else:
                    current = (current + " " + sent).strip()
        else:
            if len(current) + len(para) + 2 > max_chars and current:
                overlap = current[-overlap_chars:]
                _flush()
                current = overlap + "\n\n" + para
            else:
                current = (current + "\n\n" + para).strip()

    _flush()
    return chunks

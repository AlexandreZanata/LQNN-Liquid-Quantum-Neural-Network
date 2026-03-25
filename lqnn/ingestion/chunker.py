"""Quantum-inspired semantic chunker.

Instead of fixed-size chunks, we split on semantic boundaries (paragraphs,
sentences) and treat each chunk as a quantum state candidate. The CLIP encoder
then collapses each chunk into a 512-d vector that coexists in superposition
inside ChromaDB -- the brain queries collapse to the most relevant ones.

Strategy (based on research):
- Semantic chunking: split on paragraph/sentence boundaries
- Chunk size: ~300-500 tokens (≈1500-2500 chars) with 10% overlap
- Each chunk carries source metadata (title, author, page, url)
- Overlap ensures context continuity between adjacent chunks
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class TextChunk:
    text: str
    index: int
    source_type: str          # "pdf", "url", "text", "image"
    source_name: str          # filename or url
    metadata: dict = field(default_factory=dict)


def semantic_chunk(
    text: str,
    source_type: str,
    source_name: str,
    max_chars: int = 2000,
    overlap_chars: int = 200,
    metadata: dict | None = None,
) -> list[TextChunk]:
    """Split text into semantic chunks respecting paragraph boundaries.

    Quantum rationale: each chunk is a candidate quantum state. The overlap
    creates entanglement between adjacent chunks so context is preserved
    when the memory collapses on a query.
    """
    meta = metadata or {}
    text = text.strip()
    if not text:
        return []

    # Split on double newlines (paragraphs) first
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[TextChunk] = []
    current = ""
    idx = 0

    for para in paragraphs:
        # If a single paragraph exceeds max_chars, split by sentences
        if len(para) > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if len(current) + len(sent) + 1 > max_chars and current:
                    chunks.append(TextChunk(
                        text=current.strip(),
                        index=idx,
                        source_type=source_type,
                        source_name=source_name,
                        metadata={**meta, "chunk_index": idx},
                    ))
                    # Overlap: keep last overlap_chars of current
                    current = current[-overlap_chars:] + " " + sent
                    idx += 1
                else:
                    current = (current + " " + sent).strip()
        else:
            if len(current) + len(para) + 2 > max_chars and current:
                chunks.append(TextChunk(
                    text=current.strip(),
                    index=idx,
                    source_type=source_type,
                    source_name=source_name,
                    metadata={**meta, "chunk_index": idx},
                ))
                current = current[-overlap_chars:] + "\n\n" + para
                idx += 1
            else:
                current = (current + "\n\n" + para).strip()

    if current.strip():
        chunks.append(TextChunk(
            text=current.strip(),
            index=idx,
            source_type=source_type,
            source_name=source_name,
            metadata={**meta, "chunk_index": idx},
        ))

    return chunks

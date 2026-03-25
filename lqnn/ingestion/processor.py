"""Knowledge Ingestion Pipeline.

Orchestrates the full flow:
  1. Extract content (PDF / text / image / URL)
  2. Semantic chunking (quantum superposition candidates)
  3. JudgeAgent validation
  4. CLIP encoding + LLM association generation
  5. Store in ChromaDB with source metadata
  6. Broadcast progress events to UI

Quantum logic applied:
- Each chunk is encoded as a 512-d CLIP vector (quantum state)
- All chunks from the same document coexist in superposition
- On query, the memory collapses to the most relevant chunks
- Images from PDFs are encoded with 70/30 visual/text weighting
- Duplicate detection prevents redundant states
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from lqnn.core.associative_memory import AssociativeMemory
from lqnn.core.vector_store import VectorEntry
from lqnn.ingestion.chunker import semantic_chunk, TextChunk
from lqnn.ingestion.extractors import (
    extract_pdf, extract_text, extract_image, extract_url, ExtractedContent
)

log = logging.getLogger(__name__)

# Minimum text quality thresholds
MIN_CHUNK_CHARS = 80
MAX_CHUNK_CHARS = 3000


@dataclass
class IngestionResult:
    source: str
    source_type: str
    chunks_total: int = 0
    chunks_stored: int = 0
    chunks_rejected: int = 0
    images_stored: int = 0
    concepts_created: int = 0
    duration_s: float = 0.0
    error: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def success(self) -> bool:
        return not self.error and (self.chunks_stored > 0 or self.images_stored > 0)


HISTORY_FILE = "data/ingestion_history.json"


class KnowledgeIngestionPipeline:
    """Processes user-curated training data into the quantum memory."""

    def __init__(
        self,
        memory: AssociativeMemory,
        event_callback: Callable[[dict], None] | None = None,
    ) -> None:
        self.memory = memory
        self._emit = event_callback or (lambda e: None)
        self._history: list[dict] = self._load_history()

    def _load_history(self) -> list[dict]:
        import json, os
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, "r") as f:
                    data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            pass
        return []

    def _save_history(self) -> None:
        import json, os
        try:
            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            with open(HISTORY_FILE, "w") as f:
                json.dump(self._history[-200:], f)
        except Exception as e:
            log.debug("Failed to save ingestion history: %s", e)

    def _record_result(self, result: IngestionResult) -> None:
        entry = {
            "source": result.source,
            "source_type": result.source_type,
            "chunks_total": result.chunks_total,
            "chunks_stored": result.chunks_stored,
            "chunks_rejected": result.chunks_rejected,
            "images_stored": result.images_stored,
            "concepts_created": result.concepts_created,
            "duration_s": round(result.duration_s, 2),
            "success": result.success,
            "error": result.error,
            "timestamp": result.timestamp,
        }
        self._history.append(entry)
        if len(self._history) > 200:
            self._history = self._history[-200:]
        self._save_history()

    @property
    def history(self) -> list[dict]:
        return self._history

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def ingest_pdf(self, data: bytes, filename: str,
                         tags: list[str] | None = None) -> IngestionResult:
        """Ingest a PDF file (papers, books, articles)."""
        t0 = time.time()
        self._emit_progress("pdf", filename, "extracting", 0)

        content = await asyncio.to_thread(extract_pdf, data, filename)
        if content.error:
            return IngestionResult(source=filename, source_type="pdf",
                                   error=content.error, duration_s=time.time() - t0)

        meta = {**content.metadata, "tags": ",".join(tags or []), "type": "pdf"}
        result = await self._process_text_content(content, "pdf", filename, meta, t0)

        # Also process embedded images
        for i, img_bytes in enumerate(content.images):
            caption = content.image_captions[i] if i < len(content.image_captions) else filename
            await self._ingest_image_bytes(img_bytes, caption, filename, result)

        result.duration_s = time.time() - t0
        self._emit_done(result)
        self._record_result(result)
        return result

    async def ingest_text(self, data: bytes, filename: str,
                          tags: list[str] | None = None) -> IngestionResult:
        """Ingest plain text or markdown file."""
        t0 = time.time()
        self._emit_progress("text", filename, "extracting", 0)

        content = await asyncio.to_thread(extract_text, data, filename)
        if content.error:
            return IngestionResult(source=filename, source_type="text",
                                   error=content.error, duration_s=time.time() - t0)

        meta = {**content.metadata, "tags": ",".join(tags or []), "type": "text"}
        result = await self._process_text_content(content, "text", filename, meta, t0)
        result.duration_s = time.time() - t0
        self._emit_done(result)
        self._record_result(result)
        return result

    async def ingest_image(self, data: bytes, filename: str,
                           concept_hint: str = "",
                           tags: list[str] | None = None) -> IngestionResult:
        """Ingest an image file directly into the visual memory."""
        t0 = time.time()
        self._emit_progress("image", filename, "encoding", 0)

        content = await asyncio.to_thread(extract_image, data, filename)
        if content.error:
            return IngestionResult(source=filename, source_type="image",
                                   error=content.error, duration_s=time.time() - t0)

        result = IngestionResult(source=filename, source_type="image")
        concept = concept_hint or content.metadata.get("filename", filename)
        await self._ingest_image_bytes(content.images[0], concept, filename, result)
        result.duration_s = time.time() - t0
        self._emit_done(result)
        self._record_result(result)
        return result

    async def ingest_url(self, url: str,
                         tags: list[str] | None = None) -> IngestionResult:
        """Fetch and ingest content from a URL (article, paper, blog)."""
        t0 = time.time()
        self._emit_progress("url", url, "fetching", 0)

        content = await extract_url(url)
        if content.error:
            return IngestionResult(source=url, source_type="url",
                                   error=content.error, duration_s=time.time() - t0)

        title = content.metadata.get("title", url)
        meta = {**content.metadata, "tags": ",".join(tags or []), "type": "url"}
        result = await self._process_text_content(content, "url", title, meta, t0)
        result.duration_s = time.time() - t0
        self._emit_done(result)
        self._record_result(result)
        return result

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    async def _process_text_content(
        self,
        content: ExtractedContent,
        source_type: str,
        source_name: str,
        meta: dict,
        t0: float,
    ) -> IngestionResult:
        result = IngestionResult(source=source_name, source_type=source_type)

        if not content.text.strip():
            result.error = "No text content extracted"
            return result

        chunks = semantic_chunk(
            text=content.text,
            source_type=source_type,
            source_name=source_name,
            metadata=meta,
        )
        result.chunks_total = len(chunks)
        self._emit_progress(source_type, source_name, "chunking", 5,
                            extra={"total_chunks": len(chunks)})

        for i, chunk in enumerate(chunks):
            ok = await asyncio.to_thread(self._store_chunk, chunk)
            if ok:
                result.chunks_stored += 1
                result.concepts_created += 1
            else:
                result.chunks_rejected += 1

            # Emit progress every 5 chunks
            if i % 5 == 0:
                pct = int((i / max(len(chunks), 1)) * 90) + 5
                self._emit_progress(source_type, source_name, "encoding", pct,
                                    extra={"stored": result.chunks_stored,
                                           "rejected": result.chunks_rejected})
            # Yield control to avoid blocking the event loop
            await asyncio.sleep(0)

        return result

    def _store_chunk(self, chunk: TextChunk) -> bool:
        """Encode a text chunk and store it in the quantum memory."""
        text = chunk.text.strip()
        if len(text) < MIN_CHUNK_CHARS:
            return False

        # Quality check: must have enough alphabetic content
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.45:
            return False

        try:
            # Encode with CLIP (text path)
            vector = self.memory.clip.encode_text(text[:512])  # CLIP token limit

            # Duplicate detection
            results = self.memory.store.query_concepts(vector, n=1)
            if results and results[0].get("distance", 1.0) < 0.08:
                log.debug("Duplicate chunk skipped: %s...", text[:40])
                return False

            # Build a short concept label from the first sentence
            concept_label = text.split(".")[0][:80].strip() or text[:80]

            chunk_id = "kb_" + hashlib.md5(text.encode()).hexdigest()[:16]

            entry = VectorEntry(
                id=chunk_id,
                vector=vector,
                concept=concept_label,
                source=chunk.source_name,
                volatility=0.3,
                confidence=0.7,
                metadata={
                    "full_text": text[:2000],
                    "source_type": chunk.source_type,
                    "curation": "user_curated",
                    **{k: str(v) for k, v in chunk.metadata.items()},
                },
            )
            self.memory.store.add_concept(entry)

            # Generate associations for richer quantum superposition
            self.memory._generate_associations(concept_label, vector)

            self._emit({
                "type": "kb_chunk_stored",
                "source": chunk.source_name,
                "concept": concept_label[:50],
                "chunk_index": chunk.index,
                "timestamp": time.time(),
            })
            return True

        except Exception as e:
            log.warning("Failed to store chunk: %s", e)
            return False

    async def _ingest_image_bytes(
        self,
        img_bytes: bytes,
        concept: str,
        source_name: str,
        result: IngestionResult,
    ) -> None:
        """Encode an image and store it with visual-first weighting (70/30)."""
        try:
            state = await asyncio.to_thread(
                self.memory.learn_concept,
                concept,
                img_bytes,
                f"user_curated:kb_image:{source_name}",
                0.7,
            )
            result.images_stored += 1
            result.concepts_created += 1
            self._emit({
                "type": "kb_image_stored",
                "source": source_name,
                "concept": concept[:50],
                "associations": len(state.associations),
                "timestamp": time.time(),
            })
        except Exception as e:
            log.warning("Image ingestion failed: %s", e)

    def _emit_progress(
        self,
        source_type: str,
        source: str,
        stage: str,
        pct: int,
        extra: dict | None = None,
    ) -> None:
        event = {
            "type": "kb_progress",
            "source_type": source_type,
            "source": source[:60],
            "stage": stage,
            "percent": pct,
            "timestamp": time.time(),
        }
        if extra:
            event.update(extra)
        self._emit(event)

    def _emit_done(self, result: IngestionResult) -> None:
        self._emit({
            "type": "kb_done",
            "source": result.source[:60],
            "source_type": result.source_type,
            "chunks_total": result.chunks_total,
            "chunks_stored": result.chunks_stored,
            "chunks_rejected": result.chunks_rejected,
            "images_stored": result.images_stored,
            "concepts_created": result.concepts_created,
            "duration_s": round(result.duration_s, 2),
            "success": result.success,
            "error": result.error,
            "timestamp": time.time(),
        })

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
MIN_CHUNK_CHARS = 40
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
        self._ingested_hashes: set[str] = self._build_hash_index()

    def _load_history(self) -> list[dict]:
        import json
        import os
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, "r") as f:
                    data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            pass
        return []

    def _build_hash_index(self) -> set[str]:
        """Build a set of content hashes from successful ingestions."""
        hashes = set()
        for entry in self._history:
            h = entry.get("content_hash")
            if h and entry.get("success"):
                hashes.add(h)
        return hashes

    def _save_history(self) -> None:
        import json
        import os
        try:
            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            with open(HISTORY_FILE, "w") as f:
                json.dump(self._history[-200:], f)
        except Exception as e:
            log.debug("Failed to save ingestion history: %s", e)

    def _record_result(self, result: IngestionResult,
                       content_hash: str = "") -> None:
        existing = next(
            (e for e in self._history
             if e.get("source") == result.source and e.get("success")),
            None,
        )
        if existing and result.success:
            existing["chunks_stored"] = result.chunks_stored
            existing["chunks_total"] = result.chunks_total
            existing["chunks_rejected"] = result.chunks_rejected
            existing["images_stored"] = result.images_stored
            existing["concepts_created"] = result.concepts_created
            existing["duration_s"] = round(result.duration_s, 2)
            existing["success"] = result.success
            existing["error"] = result.error
            existing["timestamp"] = result.timestamp
            if content_hash:
                existing["content_hash"] = content_hash
            self._save_history()
            return

        if existing and not result.success:
            self._save_history()
            return

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
            "content_hash": content_hash,
        }
        self._history.append(entry)
        if len(self._history) > 200:
            self._history = self._history[-200:]
        self._save_history()
        if content_hash and result.success:
            self._ingested_hashes.add(content_hash)

    def is_already_ingested(self, data: bytes) -> str | None:
        """Return the content hash if already successfully ingested, else None."""
        h = hashlib.md5(data).hexdigest()
        return h if h in self._ingested_hashes else None

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
        content_hash = hashlib.md5(data).hexdigest()

        if content_hash in self._ingested_hashes:
            log.info("Skipping already-ingested file: %s", filename)
            existing = next(
                (e for e in self._history
                 if e.get("content_hash") == content_hash and e.get("success")),
                None,
            )
            result = IngestionResult(
                source=filename, source_type="pdf",
                chunks_stored=existing.get("chunks_stored", 0) if existing else 0,
                chunks_total=existing.get("chunks_total", 0) if existing else 0,
                concepts_created=existing.get("concepts_created", 0) if existing else 0,
                images_stored=existing.get("images_stored", 0) if existing else 0,
                duration_s=time.time() - t0,
                error="already_ingested",
            )
            self._emit_done(result)
            return result

        self._emit_progress("pdf", filename, "extracting", 0)

        content = await asyncio.to_thread(extract_pdf, data, filename)
        if content.error:
            return IngestionResult(source=filename, source_type="pdf",
                                   error=content.error, duration_s=time.time() - t0)

        meta = {**content.metadata, "tags": ",".join(tags or []), "type": "pdf"}
        result = await self._process_text_content(content, "pdf", filename, meta, t0)

        for i, img_bytes in enumerate(content.images):
            caption = content.image_captions[i] if i < len(content.image_captions) else filename
            await self._ingest_image_bytes(img_bytes, caption, filename, result)

        result.duration_s = time.time() - t0
        self._emit_done(result)
        self._record_result(result, content_hash)
        return result

    async def ingest_text(self, data: bytes, filename: str,
                          tags: list[str] | None = None) -> IngestionResult:
        """Ingest plain text or markdown file."""
        t0 = time.time()
        content_hash = hashlib.md5(data).hexdigest()

        if content_hash in self._ingested_hashes:
            log.info("Skipping already-ingested file: %s", filename)
            result = IngestionResult(
                source=filename, source_type="text",
                duration_s=time.time() - t0,
                error="already_ingested",
            )
            self._emit_done(result)
            return result

        self._emit_progress("text", filename, "extracting", 0)

        content = await asyncio.to_thread(extract_text, data, filename)
        if content.error:
            return IngestionResult(source=filename, source_type="text",
                                   error=content.error, duration_s=time.time() - t0)

        meta = {**content.metadata, "tags": ",".join(tags or []), "type": "text"}
        result = await self._process_text_content(content, "text", filename, meta, t0)
        result.duration_s = time.time() - t0
        self._emit_done(result)
        self._record_result(result, content_hash)
        return result

    async def ingest_image(self, data: bytes, filename: str,
                           concept_hint: str = "",
                           tags: list[str] | None = None) -> IngestionResult:
        """Ingest an image file directly into the visual memory."""
        t0 = time.time()
        content_hash = hashlib.md5(data).hexdigest()

        if content_hash in self._ingested_hashes:
            log.info("Skipping already-ingested image: %s", filename)
            result = IngestionResult(
                source=filename, source_type="image",
                duration_s=time.time() - t0,
                error="already_ingested",
            )
            self._emit_done(result)
            return result

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
        self._record_result(result, content_hash)
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

        raw_text = content.text.strip()
        if not raw_text:
            result.error = "No text content extracted"
            return result

        cleaned = self._clean_extracted_text(raw_text)
        log.info("Extracted %d chars from %s (cleaned to %d chars)",
                 len(raw_text), source_name, len(cleaned))

        chunks = semantic_chunk(
            text=cleaned,
            source_type=source_type,
            source_name=source_name,
            metadata=meta,
        )
        result.chunks_total = len(chunks)
        log.info("Split %s into %d chunks", source_name, len(chunks))
        self._emit_progress(source_type, source_name, "chunking", 5,
                            extra={"total_chunks": len(chunks)})

        batch_size = 10
        for i, chunk in enumerate(chunks):
            ok = await asyncio.to_thread(self._store_chunk, chunk)
            if ok:
                result.chunks_stored += 1
                result.concepts_created += 1
            else:
                result.chunks_rejected += 1

            if (i + 1) % batch_size == 0 or i == len(chunks) - 1:
                pct = int(((i + 1) / max(len(chunks), 1)) * 90) + 5
                self._emit_progress(source_type, source_name, "encoding", pct,
                                    extra={"stored": result.chunks_stored,
                                           "rejected": result.chunks_rejected,
                                           "processed": i + 1,
                                           "total": len(chunks)})
                log.info("  [%s] %d/%d chunks (stored=%d rejected=%d)",
                         source_name[:30], i + 1, len(chunks),
                         result.chunks_stored, result.chunks_rejected)
            await asyncio.sleep(0)

        return result

    @staticmethod
    def _clean_extracted_text(text: str) -> str:
        """Remove noise common in PDFs: page headers/footers, excessive
        whitespace, ligature artefacts, control characters."""
        import re
        import unicodedata
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        text = re.sub(r"\f", "\n\n", text)
        text = re.sub(r"[ \t]{3,}", "  ", text)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue
            if len(stripped) < 5 and stripped.isdigit():
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _store_chunk(self, chunk: TextChunk) -> bool:
        """Encode a text chunk and store it in the quantum memory.

        Uses CLIP-only encoding (fast, ~10ms) without triggering LLM association
        generation. Associations are deferred to the background worker thread
        so book ingestion doesn't take hours.
        """
        text = chunk.text.strip()
        if len(text) < MIN_CHUNK_CHARS:
            return False
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS]

        printable = sum(c.isprintable() for c in text) / max(len(text), 1)
        if printable < 0.50:
            return False

        try:
            vector = self.memory._cached_encode_text(text[:512])

            try:
                results = self.memory.store.query_concepts(vector, n=1)
                if results and results[0].get("distance", 1.0) < 0.08:
                    log.debug("Duplicate chunk skipped: %s...", text[:40])
                    return False
            except Exception:
                pass

            concept_label = self._extract_concept_label(text)
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

            # Queue 5 associations for background generation instead of
            # blocking with a full 30-association LLM call per chunk
            import queue as _q
            try:
                self.memory._assoc_bg_queue.put_nowait(
                    (concept_label, vector, 5))
            except _q.Full:
                pass

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

    @staticmethod
    def _extract_concept_label(text: str) -> str:
        """Build a meaningful concept label from the text."""
        for sep in (".", ":", "\n"):
            parts = text.split(sep, 1)
            label = parts[0].strip()
            if 10 < len(label) < 120:
                return label
        return text[:100].strip()

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

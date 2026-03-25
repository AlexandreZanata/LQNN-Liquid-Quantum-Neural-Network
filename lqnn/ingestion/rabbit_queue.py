from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path

import aio_pika
from aio_pika import DeliveryMode, Message

log = logging.getLogger(__name__)

QUEUE_NAME = "lqnn_kb_ingestion"
QUEUE_FILES_DIR = Path("data/queue_files")


class RabbitIngestionQueue:
    def __init__(self, rabbitmq_url: str, ingestion_pipeline, event_callback=None) -> None:
        self.rabbitmq_url = rabbitmq_url
        self.ingestion = ingestion_pipeline
        self._emit = event_callback or (lambda e: None)
        self._connection = None
        self._channel = None
        self._queue = None
        self._consumer_tag = None

    async def start(self) -> None:
        QUEUE_FILES_DIR.mkdir(parents=True, exist_ok=True)
        self._connection = await aio_pika.connect_robust(self.rabbitmq_url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=1)
        self._queue = await self._channel.declare_queue(QUEUE_NAME, durable=True)
        self._consumer_tag = await self._queue.consume(
            self._on_message, no_ack=False)
        log.info("Rabbit ingestion queue ready: %s", QUEUE_NAME)

    async def stop(self) -> None:
        if self._queue and self._consumer_tag:
            await self._queue.cancel(self._consumer_tag)
        if self._channel and not self._channel.is_closed:
            await self._channel.close()
        if self._connection and not self._connection.is_closed:
            await self._connection.close()

    async def enqueue_file(
        self,
        data: bytes,
        filename: str,
        tags: list[str] | None = None,
        concept_hint: str = "",
    ) -> dict:
        QUEUE_FILES_DIR.mkdir(parents=True, exist_ok=True)
        file_id = f"{int(time.time())}_{uuid.uuid4().hex}"
        file_path = QUEUE_FILES_DIR / f"{file_id}.bin"
        file_path.write_bytes(data)

        payload = {
            "kind": "file",
            "file_path": str(file_path),
            "filename": filename,
            "tags": tags or [],
            "concept_hint": concept_hint or "",
            "queued_at": time.time(),
        }
        await self._publish(payload)
        self._emit({
            "type": "kb_queued",
            "source": filename,
            "source_type": "file",
            "timestamp": time.time(),
        })
        return {"queued": True, "source": filename, "queue_id": file_id}

    async def enqueue_url(self, url: str, tags: list[str] | None = None) -> dict:
        payload = {
            "kind": "url",
            "url": url,
            "tags": tags or [],
            "queued_at": time.time(),
        }
        await self._publish(payload)
        self._emit({
            "type": "kb_queued",
            "source": url,
            "source_type": "url",
            "timestamp": time.time(),
        })
        return {"queued": True, "source": url}

    async def _publish(self, payload: dict) -> None:
        if not self._channel:
            raise RuntimeError("Rabbit queue not started")
        body = json.dumps(payload).encode("utf-8")
        await self._channel.default_exchange.publish(
            Message(body=body, delivery_mode=DeliveryMode.PERSISTENT),
            routing_key=QUEUE_NAME,
        )

    async def _on_message(self, message: aio_pika.IncomingMessage) -> None:
        payload: dict = {}
        source = "unknown"
        try:
            payload = json.loads(message.body.decode("utf-8"))
            kind = payload.get("kind")
            source = payload.get("filename", payload.get("url", "unknown"))
            log.info("Processing ingestion job: %s [%s]", kind, source)

            if kind == "file":
                await self._consume_file(payload)
            elif kind == "url":
                await self._consume_url(payload)
            else:
                log.warning("Unknown ingestion job kind: %s", kind)

            await message.ack()
            log.info("Ingestion job completed: %s [%s]", kind, source)

        except Exception as exc:
            log.error("Ingestion job FAILED for %s: %s",
                      source, exc, exc_info=True)
            self._emit({
                "type": "kb_error",
                "source": source,
                "error": str(exc)[:200],
                "timestamp": time.time(),
            })
            try:
                await message.reject(requeue=False)
            except Exception:
                pass

    async def _consume_file(self, payload: dict) -> None:
        path = payload.get("file_path", "")
        filename = payload.get("filename", "upload")
        tags = payload.get("tags", [])
        concept_hint = payload.get("concept_hint", "")

        if not path or not os.path.exists(path):
            raise RuntimeError(f"Queued file missing: {path}")

        data = Path(path).read_bytes()
        log.info("Ingesting file: %s (%d bytes)", filename, len(data))
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if ext == "pdf":
            result = await self.ingestion.ingest_pdf(data, filename, tags)
        elif ext in {"jpg", "jpeg", "png", "webp", "gif", "bmp"}:
            result = await self.ingestion.ingest_image(data, filename, concept_hint, tags)
        elif ext in {"html", "htm"}:
            result = await self.ingestion.ingest_text(data, filename, tags)
        elif ext in {"docx", "doc"}:
            result = await self.ingestion.ingest_text(data, filename, tags)
        else:
            result = await self.ingestion.ingest_text(data, filename, tags)

        log.info("Ingestion result for %s: stored=%d rejected=%d images=%d error=%s",
                 filename, result.chunks_stored, result.chunks_rejected,
                 result.images_stored, result.error or "none")

        try:
            os.remove(path)
        except OSError:
            pass

    async def _consume_url(self, payload: dict) -> None:
        url = payload.get("url", "")
        tags = payload.get("tags", [])
        if not url:
            raise RuntimeError("Queued url job missing URL")
        log.info("Ingesting URL: %s", url)
        result = await self.ingestion.ingest_url(url, tags)
        log.info("URL ingestion result for %s: stored=%d error=%s",
                 url[:60], result.chunks_stored, result.error or "none")

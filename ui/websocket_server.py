"""WebSocket server for real-time UI updates."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from ui.controls import UIController

log = logging.getLogger(__name__)


class WebSocketStateServer:
    """Pushes brain state + live events to connected UI clients."""

    FPS = 2

    def __init__(self, controller: UIController) -> None:
        self.controller = controller
        self._clients: set[WebSocket] = set()
        self._task: asyncio.Task | None = None
        self._running = False
        self._event_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=500)

    def push_event(self, event: dict) -> None:
        """Push a real-time event (training/agent) to all clients."""
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._broadcast_loop())
        asyncio.create_task(self._event_forwarder())
        asyncio.create_task(self._reactive_learning_consumer())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)
        try:
            snapshot = self.controller.snapshot()
            await ws.send_json(snapshot)
        except Exception:
            pass

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def _broadcast_to_all(self, data: dict) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def _event_forwarder(self) -> None:
        """Forward real-time events from the queue to all clients."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=1.0)
                if self._clients:
                    await self._broadcast_to_all({
                        "type": "live_event",
                        **event,
                    })
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def _reactive_learning_consumer(self) -> None:
        """Process reactive learning requests from the chat engine."""
        reactive_queue = getattr(self.controller, 'reactive_queue', None)
        if reactive_queue is None:
            return

        while self._running:
            try:
                topic = await asyncio.wait_for(reactive_queue.get(), timeout=2.0)
                if topic and self.controller.agent_manager:
                    try:
                        report = await self.controller.agent_manager.request_search(topic)
                        log.info("Reactive learning completed for '%s': %d concepts",
                                 topic[:40], report.concepts_learned)
                    except Exception as e:
                        log.warning("Reactive learning failed for '%s': %s", topic[:40], e)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def handle_message(self, ws: WebSocket, data: dict[str, Any]) -> None:
        action = data.get("action", "")
        try:
            if action == "chat":
                result = await asyncio.to_thread(
                    self.controller.chat_turn, data.get("text", "")
                )
                await ws.send_json({"type": "chat_response", **result})

            elif action == "search":
                result = await self.controller.search(data.get("query", ""))
                await ws.send_json({"type": "search_result", **result})

            elif action == "learn":
                result = await asyncio.to_thread(
                    self.controller.learn_concept, data.get("concept", "")
                )
                await ws.send_json({"type": "learn_result", **result})

            elif action == "consolidate":
                result = await asyncio.to_thread(self.controller.consolidate)
                await ws.send_json({"type": "consolidation_result", **result})

            elif action == "self_play":
                result = await asyncio.to_thread(self.controller.self_play)
                await ws.send_json({"type": "self_play_result", **result})

            elif action == "train_cycle":
                result = await self.controller.train_cycle()
                await ws.send_json({"type": "train_result", **result})

            elif action == "agent_cycle":
                result = await self.controller.run_agent_cycle()
                await ws.send_json({"type": "agent_result", **result})

            elif action == "start_training":
                await self.controller.trainer.start()
                await ws.send_json({"type": "training_started"})

            elif action == "stop_training":
                await self.controller.trainer.stop()
                await ws.send_json({"type": "training_stopped"})

            elif action == "kb_ingest_url":
                url = data.get("url", "")
                tags = data.get("tags", [])
                if not url:
                    await ws.send_json({"type": "error", "message": "url required"})
                elif not self.controller.ingestion:
                    await ws.send_json({"type": "error", "message": "ingestion not ready"})
                else:
                    result = await self.controller.ingestion.ingest_url(url, tags)
                    await ws.send_json({
                        "type": "kb_ingest_result",
                        "source": result.source,
                        "source_type": result.source_type,
                        "chunks_stored": result.chunks_stored,
                        "chunks_total": result.chunks_total,
                        "chunks_rejected": result.chunks_rejected,
                        "images_stored": result.images_stored,
                        "concepts_created": result.concepts_created,
                        "duration_s": round(result.duration_s, 2),
                        "success": result.success,
                        "error": result.error,
                    })

            elif action == "cleanup":
                result = await asyncio.to_thread(
                    self.controller.memory.cleanup_garbage)
                await ws.send_json({"type": "cleanup_result", **result})

            elif action == "benchmark":
                result = await asyncio.to_thread(
                    self.controller.trainer.run_benchmark)
                await ws.send_json({"type": "benchmark_result", **result})

            elif action == "get_state":
                snapshot = self.controller.snapshot()
                await ws.send_json(snapshot)

            else:
                await ws.send_json({"type": "error", "message": f"Unknown action: {action}"})

        except Exception as exc:
            log.error("WS action '%s' failed: %s", action, exc, exc_info=True)
            await ws.send_json({"type": "error", "message": str(exc)})

    async def _broadcast_loop(self) -> None:
        while self._running:
            if self._clients:
                try:
                    snapshot = self.controller.snapshot()
                    await self._broadcast_to_all(snapshot)
                except Exception as exc:
                    log.error("Broadcast error: %s", exc)

            await asyncio.sleep(1.0 / self.FPS)


def register_websocket_routes(app, server: WebSocketStateServer) -> None:
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await server.connect(ws)
        try:
            while True:
                text = await ws.receive_text()
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})
                    continue
                await server.handle_message(ws, data)
        except WebSocketDisconnect:
            server.disconnect(ws)
        except Exception:
            server.disconnect(ws)

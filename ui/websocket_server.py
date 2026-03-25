"""WebSocket server for real-time UI updates."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from ui.controls import UIController

log = logging.getLogger(__name__)

_language_trainer = None


def set_language_trainer(trainer) -> None:
    global _language_trainer
    _language_trainer = trainer


class WebSocketStateServer:
    """Pushes brain state + live events to connected UI clients."""

    FPS = 2

    def __init__(self, controller: UIController) -> None:
        self.controller = controller
        self._clients: set[WebSocket] = set()
        self._task: asyncio.Task | None = None
        self._running = False
        self._event_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=500)
        self._active_stream_tasks: dict[WebSocket, asyncio.Task] = {}

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
        task = self._active_stream_tasks.pop(ws, None)
        if task and not task.done():
            self.controller.chat_engine.cancel_active()
            log.info("Auto-cancelled active stream on WebSocket disconnect")

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
                    inner_type = event.pop("type", "unknown")
                    await self._broadcast_to_all({
                        "type": "live_event",
                        "event_type": inner_type,
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

            elif action == "chat_stream":
                text = data.get("text", "")
                session_id = data.get("session_id", "")

                async def _run_stream() -> None:
                    loop = asyncio.get_event_loop()
                    msg_queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
                    ws_closed = False

                    def _on_token(chunk: str) -> None:
                        nonlocal ws_closed
                        if not ws_closed:
                            loop.call_soon_threadsafe(
                                msg_queue.put_nowait, ("token", chunk))

                    def _on_reasoning(step: str) -> None:
                        nonlocal ws_closed
                        if not ws_closed:
                            loop.call_soon_threadsafe(
                                msg_queue.put_nowait, ("reasoning", step))

                    async def _send_messages() -> None:
                        nonlocal ws_closed
                        while True:
                            item = await msg_queue.get()
                            if item is None:
                                break
                            kind, val = item
                            try:
                                if ws.client_state != WebSocketState.CONNECTED:
                                    ws_closed = True
                                    break
                                if kind == "token":
                                    await ws.send_json(
                                        {"type": "chat_token", "token": val})
                                elif kind == "reasoning":
                                    await ws.send_json(
                                        {"type": "chat_reasoning", "step": val})
                            except Exception:
                                ws_closed = True
                                break

                    sender_task = asyncio.create_task(_send_messages())
                    result = await asyncio.to_thread(
                        self.controller.chat_engine.chat_stream,
                        text, _on_token, _on_reasoning,
                    )
                    loop.call_soon_threadsafe(msg_queue.put_nowait, None)
                    await sender_task

                    self._persist_stream_result(session_id, text, result)

                    is_cancelled = result.get("status") == "cancelled"

                    if not ws_closed and ws.client_state == WebSocketState.CONNECTED:
                        try:
                            if is_cancelled:
                                await ws.send_json({"type": "stream_cancelled"})
                            else:
                                await ws.send_json(
                                    {"type": "chat_response", **result})
                        except Exception:
                            log.debug("WS closed before final response sent")
                    else:
                        log.info(
                            "WS disconnected mid-stream; response saved "
                            "to session %s",
                            session_id[:12] if session_id else "(none)")

                    self._active_stream_tasks.pop(ws, None)

                stream_task = asyncio.create_task(_run_stream())
                self._active_stream_tasks[ws] = stream_task

            elif action == "cancel_stream":
                cancelled = self.controller.chat_engine.cancel_active()
                task = self._active_stream_tasks.get(ws)
                if task and not task.done():
                    log.info("Stream cancel requested by client")
                if not cancelled:
                    try:
                        if ws.client_state == WebSocketState.CONNECTED:
                            await ws.send_json({"type": "stream_cancelled"})
                    except Exception:
                        pass

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
                elif not getattr(self.controller, "ingestion_queue", None):
                    await ws.send_json({"type": "error", "message": "ingestion queue not ready"})
                else:
                    result = await self.controller.ingestion_queue.enqueue_url(url, tags)
                    await ws.send_json({
                        "type": "kb_queue_result",
                        "source": result.get("source", url),
                        "queued": True,
                    })

            elif action == "cleanup":
                result = await asyncio.to_thread(
                    self.controller.memory.cleanup_garbage)
                await ws.send_json({"type": "cleanup_result", **result})

            elif action == "benchmark":
                result = await asyncio.to_thread(
                    self.controller.trainer.run_benchmark)
                await ws.send_json({"type": "benchmark_result", **result})

            elif action == "benchmark_frontier":
                result = await asyncio.to_thread(
                    self.controller.trainer.run_frontier_benchmark)
                await ws.send_json({"type": "benchmark_frontier_result", **result})

            elif action == "get_state":
                snapshot = self.controller.snapshot()
                await ws.send_json(snapshot)

            elif action == "get_language_status":
                if _language_trainer:
                    status = _language_trainer.status()
                    await ws.send_json({"type": "language_status", **status})
                else:
                    await ws.send_json({"type": "language_status", "languages": {}})

            elif action == "start_language_training":
                if not _language_trainer:
                    await ws.send_json({"type": "error", "message": "language trainer not initialized"})
                else:
                    langs = data.get("languages")
                    await _language_trainer.start(langs)
                    await ws.send_json({"type": "language_training_started", "languages": langs})

            elif action == "stop_language_training":
                if not _language_trainer:
                    await ws.send_json({"type": "error", "message": "language trainer not initialized"})
                else:
                    langs = data.get("languages")
                    await _language_trainer.stop(langs)
                    await ws.send_json({"type": "language_training_stopped", "languages": langs})

            elif action == "download_dataset":
                if not _language_trainer:
                    await ws.send_json({"type": "error", "message": "language trainer not initialized"})
                else:
                    dataset_id = data.get("dataset_id", "")
                    result = await _language_trainer.download_dataset_manual(dataset_id)
                    await ws.send_json({"type": "dataset_download_result", **result})

            else:
                await ws.send_json({"type": "error", "message": f"Unknown action: {action}"})

        except Exception as exc:
            log.error("WS action '%s' failed: %s", action, exc, exc_info=True)
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass

    def _persist_stream_result(self, session_id: str, user_text: str,
                               result: dict) -> None:
        """Save the assistant response to the session store server-side.

        This ensures the response survives even if the WebSocket is closed
        before the frontend can save it (e.g. user reloads mid-stream).
        """
        if not session_id:
            return
        try:
            from ui.app import _chat_sessions
            if not _chat_sessions:
                return
            sess = _chat_sessions.get_session(session_id)
            if not sess:
                return
            messages = list(sess.get("messages", []))
            import time
            now_ms = int(time.time() * 1000)
            has_user_msg = any(
                m.get("role") == "user" and m.get("text") == user_text
                for m in messages[-3:]
            )
            if not has_user_msg:
                messages.append({
                    "role": "user", "text": user_text,
                    "meta": {}, "timestamp": now_ms,
                })
            messages.append({
                "role": "assistant",
                "text": result.get("response", ""),
                "meta": {
                    "confidence": result.get("confidence"),
                    "concepts": result.get("concepts"),
                    "duration_ms": result.get("duration_ms"),
                },
                "timestamp": now_ms,
            })
            _chat_sessions.upsert_session(session_id, messages=messages)
        except Exception as exc:
            log.warning("Failed to persist stream result: %s", exc)

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

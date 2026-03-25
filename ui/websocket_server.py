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
    """Pushes brain state to connected UI clients at regular intervals."""

    FPS = 2

    def __init__(self, controller: UIController) -> None:
        self.controller = controller
        self._clients: set[WebSocket] = set()
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._broadcast_loop())

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

    async def handle_message(self, ws: WebSocket, data: dict[str, Any]) -> None:
        action = data.get("action", "")
        try:
            if action == "chat":
                result = self.controller.chat_turn(data.get("text", ""))
                await ws.send_json({"type": "chat_response", **result})

            elif action == "search":
                result = await self.controller.search(data.get("query", ""))
                await ws.send_json({"type": "search_result", **result})

            elif action == "learn":
                result = self.controller.learn_concept(data.get("concept", ""))
                await ws.send_json({"type": "learn_result", **result})

            elif action == "consolidate":
                result = self.controller.consolidate()
                await ws.send_json({"type": "consolidation_result", **result})

            elif action == "self_play":
                result = self.controller.self_play()
                await ws.send_json({"type": "self_play_result", **result})

            elif action == "train_cycle":
                result = await self.controller.train_cycle()
                await ws.send_json({"type": "train_result", **result})

            elif action == "agent_cycle":
                result = await self.controller.run_agent_cycle()
                await ws.send_json({"type": "agent_result", **result})

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
                    dead: list[WebSocket] = []
                    for ws in list(self._clients):
                        try:
                            await ws.send_json(snapshot)
                        except Exception:
                            dead.append(ws)
                    for ws in dead:
                        self._clients.discard(ws)
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

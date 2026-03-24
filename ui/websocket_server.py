# Systems principle: a continuous event loop streams true internal state to
# clients, enabling real-time observation of structural adaptation.

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ui.controls import UIController


class WebSocketStateServer:
    """Streams live network state and handles control messages."""

    def __init__(self, controller: UIController, fps: int = 15) -> None:
        self.controller = controller
        self.fps = max(10, min(20, fps))
        self._clients: set[WebSocket] = set()
        self._task: asyncio.Task[Any] | None = None
        self._running = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._broadcast_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        await websocket.send_json(self.controller.snapshot())

    async def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._clients:
            self._clients.remove(websocket)

    async def handle_message(self, websocket: WebSocket, raw: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send_json({"type": "error", "message": "JSON invalido"})
            return

        action = message.get("action", "")
        async with self._lock:
            if action == "stimulate":
                text = str(message.get("text", ""))
                state = self.controller.stimulate_text(text) if text else self.controller.stimulate_random()
            elif action == "stimulate_vector":
                state = self.controller.stimulate_vector(message.get("values", []))
            elif action == "sleep":
                state = self.controller.sleep(int(message.get("cycles", 1)))
            elif action == "reset":
                state = self.controller.reset_network()
            elif action == "toggle_quantum":
                enabled = message.get("enabled")
                state = self.controller.toggle_quantum_mode(enabled if isinstance(enabled, bool) else None)
            elif action == "toggle_auto":
                state = self.controller.set_auto_stimulate(bool(message.get("enabled", True)))
            elif action == "get_state":
                state = self.controller.snapshot()
            else:
                await websocket.send_json({"type": "error", "message": f"Acao desconhecida: {action}"})
                return
        await websocket.send_json(state)

    async def _broadcast_loop(self) -> None:
        frame_time = 1.0 / float(self.fps)
        while self._running:
            async with self._lock:
                payload = self.controller.idle_step()

            to_remove: list[WebSocket] = []
            for client in self._clients:
                try:
                    await client.send_json(payload)
                except Exception:
                    to_remove.append(client)

            for client in to_remove:
                await self.disconnect(client)

            await asyncio.sleep(frame_time)


def register_websocket_routes(app: FastAPI, server: WebSocketStateServer) -> None:
    """Register startup/shutdown hooks and websocket endpoint."""

    @app.on_event("startup")
    async def _startup() -> None:
        await server.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await server.stop()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await server.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                await server.handle_message(websocket, data)
        except WebSocketDisconnect:
            await server.disconnect(websocket)

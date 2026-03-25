"""FastAPI application -- serves the LQNN v2 web interface."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

log = logging.getLogger(__name__)

_controller = None
_ws_server = None
_trainer = None


class ChatRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    query: str


class LearnRequest(BaseModel):
    concept: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _ws_server:
        await _ws_server.start()
    if _trainer:
        await _trainer.start()
    yield
    if _trainer:
        await _trainer.stop()
    if _ws_server:
        await _ws_server.stop()


def create_app(controller=None, ws_server=None, trainer=None) -> FastAPI:
    global _controller, _ws_server, _trainer
    _controller = controller
    _ws_server = ws_server
    _trainer = trainer

    app = FastAPI(title="LQNN v2 - Quantum Associative Brain", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory="ui/static"), name="static")

    if ws_server:
        from ui.websocket_server import register_websocket_routes
        register_websocket_routes(app, ws_server)

    # -- Pages --

    @app.get("/")
    async def root():
        return RedirectResponse("/chat")

    @app.get("/chat")
    async def chat_page():
        return FileResponse("ui/static/chat.html")

    @app.get("/training")
    async def training_page():
        return FileResponse("ui/static/training.html")

    # -- Health --

    @app.get("/health")
    async def health():
        return {"status": "alive", "version": "2.0.0"}

    # -- Chat API --

    @app.post("/api/chat")
    async def api_chat(req: ChatRequest):
        if not _controller:
            return {"error": "not_initialized"}
        return _controller.chat_turn(req.text)

    # -- Search API --

    @app.post("/api/search")
    async def api_search(req: SearchRequest):
        if not _controller:
            return {"error": "not_initialized"}
        return await _controller.search(req.query)

    # -- Learn API --

    @app.post("/api/learn")
    async def api_learn(req: LearnRequest):
        if not _controller:
            return {"error": "not_initialized"}
        return _controller.learn_concept(req.concept)

    # -- Memory / Brain Status --

    @app.get("/api/brain/status")
    async def brain_status():
        if not _controller:
            return {"error": "not_initialized"}
        return _controller.snapshot()

    @app.get("/api/memory/stats")
    async def memory_stats():
        if not _controller:
            return {"error": "not_initialized"}
        return _controller.memory.stats()

    # -- Training --

    @app.get("/api/training/status")
    async def training_status():
        if not _trainer:
            return {"error": "not_initialized"}
        return _trainer.status()

    @app.post("/api/training/cycle")
    async def training_cycle():
        if not _controller:
            return {"error": "not_initialized"}
        return await _controller.train_cycle()

    # -- Agents --

    @app.get("/api/agents/status")
    async def agent_status():
        if not _controller:
            return {"error": "not_initialized"}
        return _controller.agent_manager.stats()

    @app.post("/api/agents/cycle")
    async def agent_cycle():
        if not _controller:
            return {"error": "not_initialized"}
        return await _controller.run_agent_cycle()

    # -- Consolidation --

    @app.post("/api/consolidate")
    async def consolidate():
        if not _controller:
            return {"error": "not_initialized"}
        return _controller.consolidate()

    # -- Self-play --

    @app.post("/api/self-play")
    async def self_play():
        if not _controller:
            return {"error": "not_initialized"}
        return _controller.self_play()

    return app

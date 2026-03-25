"""FastAPI application -- serves the LQNN v3 web interface."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

log = logging.getLogger(__name__)

_controller = None
_ws_server = None
_trainer = None
_chat_sessions = None


class ChatRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    query: str


class LearnRequest(BaseModel):
    concept: str


class KnowledgeUrlRequest(BaseModel):
    url: str
    tags: list[str] = []


class ChatSessionCreateRequest(BaseModel):
    title: str = "New chat"


class ChatSessionSaveRequest(BaseModel):
    title: str | None = None
    messages: list[dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _controller and getattr(_controller, "ingestion_queue", None):
        await _controller.ingestion_queue.start()
    if _ws_server:
        await _ws_server.start()
    auto_train = os.environ.get("AUTO_TRAIN_ON_START", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if _trainer and auto_train:
        await _trainer.start()
    yield
    if _trainer:
        await _trainer.stop()
    if _ws_server:
        await _ws_server.stop()
    if _controller and getattr(_controller, "ingestion_queue", None):
        await _controller.ingestion_queue.stop()


def create_app(controller=None, ws_server=None, trainer=None) -> FastAPI:
    global _controller, _ws_server, _trainer, _chat_sessions
    _controller = controller
    _ws_server = ws_server
    _trainer = trainer
    from lqnn.system.chat_sessions import ChatSessionStore
    _chat_sessions = ChatSessionStore()

    app = FastAPI(title="LQNN v3 - Quantum Associative Brain", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory="ui/static"), name="static")

    if ws_server:
        from ui.websocket_server import register_websocket_routes
        register_websocket_routes(app, ws_server)

    # -- Pages (unified single-page terminal) --

    @app.get("/")
    async def root():
        return RedirectResponse("/terminal")

    @app.get("/terminal")
    async def terminal_page():
        return FileResponse("ui/static/terminal.html")

    @app.get("/chat")
    async def chat_page():
        return RedirectResponse("/terminal")

    @app.get("/training")
    async def training_page():
        return RedirectResponse("/terminal")

    # -- Health --

    @app.get("/health")
    async def health():
        status = "alive"
        if _controller and _controller.memory.llm.ready:
            status = "ready"
        elif _controller and _controller.memory.llm.loading:
            status = "loading"
        return {"status": status, "version": "3.0.0"}

    # -- System Metrics --

    @app.get("/api/system/metrics")
    async def system_metrics():
        from ui.controls import _get_system_metrics
        return _get_system_metrics()

    # -- Chat API --

    @app.post("/api/chat")
    async def api_chat(req: ChatRequest):
        if not _controller:
            return {"error": "not_initialized"}
        return await asyncio.to_thread(_controller.chat_turn, req.text)

    @app.get("/api/chat/sessions")
    async def list_chat_sessions():
        if not _chat_sessions:
            return []
        return _chat_sessions.list_sessions()

    @app.post("/api/chat/sessions")
    async def create_chat_session(req: ChatSessionCreateRequest):
        if not _chat_sessions:
            return {"error": "sessions_not_initialized"}
        return _chat_sessions.create_session(req.title)

    @app.get("/api/chat/sessions/{session_id}")
    async def get_chat_session(session_id: str):
        if not _chat_sessions:
            return {"error": "sessions_not_initialized"}
        sess = _chat_sessions.get_session(session_id)
        if not sess:
            return {"error": "not_found"}
        return sess

    @app.put("/api/chat/sessions/{session_id}")
    async def save_chat_session(session_id: str, req: ChatSessionSaveRequest):
        if not _chat_sessions:
            return {"error": "sessions_not_initialized"}
        return _chat_sessions.upsert_session(
            session_id=session_id,
            title=req.title,
            messages=req.messages,
        )

    @app.delete("/api/chat/sessions/{session_id}")
    async def delete_chat_session(session_id: str):
        if not _chat_sessions:
            return {"error": "sessions_not_initialized"}
        ok = _chat_sessions.delete_session(session_id)
        return {"ok": ok}

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
        return await asyncio.to_thread(_controller.learn_concept, req.concept)

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

    @app.post("/api/training/start")
    async def training_start():
        if not _trainer:
            return {"error": "not_initialized"}
        await _trainer.start()
        return _trainer.status()

    @app.post("/api/training/stop")
    async def training_stop():
        if not _trainer:
            return {"error": "not_initialized"}
        await _trainer.stop()
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
        return await asyncio.to_thread(_controller.consolidate)

    # -- Self-play --

    @app.post("/api/self-play")
    async def self_play():
        if not _controller:
            return {"error": "not_initialized"}
        return await asyncio.to_thread(_controller.self_play)

    # -- Knowledge Base (data upload) --

    @app.get("/knowledge")
    async def knowledge_page():
        return FileResponse("ui/static/knowledge.html")

    @app.post("/api/knowledge/upload")
    async def knowledge_upload(
        file: UploadFile = File(...),
        tags: str = "",
        concept_hint: str = "",
    ):
        if not _controller or not _controller.ingestion:
            return {"error": "not_initialized"}

        if not getattr(_controller, "ingestion_queue", None):
            return {"error": "ingestion_queue_not_initialized"}

        data = await file.read()
        fname = file.filename or "upload"
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        queued = await _controller.ingestion_queue.enqueue_file(
            data=data,
            filename=fname,
            tags=tag_list,
            concept_hint=concept_hint,
        )
        return {"queued": True, **queued}

    @app.get("/api/knowledge/history")
    async def knowledge_history():
        if not _controller or not _controller.ingestion:
            return []
        return _controller.ingestion.history

    @app.post("/api/knowledge/url")
    async def knowledge_url(req: KnowledgeUrlRequest):
        if not _controller or not _controller.ingestion:
            return {"error": "not_initialized"}
        if not getattr(_controller, "ingestion_queue", None):
            return {"error": "ingestion_queue_not_initialized"}
        queued = await _controller.ingestion_queue.enqueue_url(req.url, req.tags)
        return {"queued": True, **queued}

    @app.post("/api/cleanup")
    async def cleanup_garbage():
        if not _controller:
            return {"error": "not_initialized"}
        return await asyncio.to_thread(_controller.memory.cleanup_garbage)

    @app.get("/api/benchmark")
    async def run_benchmark():
        if not _trainer:
            return {"error": "not_initialized"}
        return await asyncio.to_thread(_trainer.run_benchmark)

    return app

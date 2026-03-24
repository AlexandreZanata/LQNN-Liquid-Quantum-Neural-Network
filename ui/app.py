# System principle: a web interface exposes the living topology as it evolves,
# so experimentation is immediate, observable, and interactive.

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ui.controls import UIController
from ui.websocket_server import WebSocketStateServer, register_websocket_routes


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="LQNN Live UI", version="0.1.0")

    controller = UIController(max_neurons=300)
    server = WebSocketStateServer(controller=controller, fps=15)
    register_websocket_routes(app, server)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ui.app:app", host="0.0.0.0", port=8000, reload=False)

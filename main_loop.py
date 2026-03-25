"""LQNN v2 -- Main entry point.

Starts:
1. The AI models (CLIP + Phi-3.5)
2. ChromaDB vector store
3. MongoDB logging
4. Continuous training loop (background)
5. FastAPI web UI (foreground)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("lqnn")


def build_system():
    """Initialize all system components."""
    log.info("=== LQNN v2 - Quantum Associative Brain ===")
    log.info("Initializing system components...")

    from lqnn.core.vector_store import VectorStore
    from lqnn.core.associative_memory import AssociativeMemory
    from lqnn.models.clip_encoder import CLIPEncoder
    from lqnn.models.llm_engine import LLMEngine
    from lqnn.agents.browser_agent import BrowserAgent
    from lqnn.agents.manager import AgentManager
    from lqnn.system.training_db import TrainingDB
    from lqnn.system.chat_engine import ChatEngine
    from lqnn.training.continuous_trainer import ContinuousTrainer
    from ui.controls import UIController
    from ui.websocket_server import WebSocketStateServer
    from ui.app import create_app

    log.info("Loading AI models (this may download on first run)...")
    clip = CLIPEncoder()
    llm = LLMEngine()

    log.info("Initializing vector store (ChromaDB)...")
    chroma_dir = os.environ.get("CHROMA_DIR", "data/chroma")
    os.makedirs(chroma_dir, exist_ok=True)
    store = VectorStore(persist_dir=chroma_dir)

    log.info("Initializing associative memory...")
    memory = AssociativeMemory(store=store, clip=clip, llm=llm)

    log.info("Connecting to MongoDB...")
    training_db = TrainingDB()

    log.info("Setting up agents...")
    browser = BrowserAgent()
    agent_manager = AgentManager(memory=memory, browser=browser)

    log.info("Setting up chat engine...")
    chat_engine = ChatEngine(memory=memory, llm=llm, training_db=training_db)

    log.info("Setting up continuous trainer...")
    trainer = ContinuousTrainer(
        memory=memory,
        agent_manager=agent_manager,
        training_db=training_db,
    )

    log.info("Setting up UI controller...")
    controller = UIController(
        memory=memory,
        chat_engine=chat_engine,
        trainer=trainer,
        agent_manager=agent_manager,
    )

    ws_server = WebSocketStateServer(controller)
    app = create_app(controller=controller, ws_server=ws_server, trainer=trainer)

    log.info("System initialized. Starting server...")
    return app


app = build_system()

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "main_loop:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )

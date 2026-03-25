"""LQNN v3 -- Main entry point.

Starts:
1. The AI models (CLIP + Qwen2.5-7B)
2. ChromaDB vector store
3. MongoDB logging
4. Quantum processing subsystems (HEI, batch engine, temporal pipeline)
5. Continuous training loop (background)
6. FastAPI web UI with hacker terminal (foreground)
"""

from __future__ import annotations

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


def _ensure_dir_writable(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    probe = os.path.join(path, ".write_probe")
    try:
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return
    except Exception:
        pass

    for root, dirs, files in os.walk(path):
        for d in dirs:
            try:
                os.chmod(os.path.join(root, d), 0o775)
            except Exception:
                pass
        for fn in files:
            try:
                os.chmod(os.path.join(root, fn), 0o664)
            except Exception:
                pass
    try:
        os.chmod(path, 0o775)
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
    except Exception as exc:
        raise RuntimeError(
            f"Chroma path is not writable: {path}. "
            f"Fix permissions for ./data (host) and restart."
        ) from exc


def build_system():
    """Initialize all system components."""
    log.info("=== LQNN v3 - Quantum Associative Brain ===")
    log.info("Initializing system components...")

    from lqnn.core.vector_store import VectorStore
    from lqnn.core.associative_memory import AssociativeMemory
    from lqnn.core.entanglement_index import EntanglementIndex
    from lqnn.core.quantum_batch_engine import QuantumBatchEngine
    from lqnn.core.temporal_pipeline import TemporalPipeline
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
    _ensure_dir_writable(chroma_dir)
    store = VectorStore(persist_dir=chroma_dir)

    log.info("Initializing associative memory...")
    memory = AssociativeMemory(store=store, clip=clip, llm=llm)

    log.info("Initializing Hierarchical Entanglement Index...")
    hei = EntanglementIndex()
    memory.set_hei(hei)

    log.info("Initializing Quantum Batch Engine...")
    batch_engine = QuantumBatchEngine(clip=clip, store=store)
    memory.set_batch_engine(batch_engine)

    log.info("Connecting to MongoDB...")
    training_db = TrainingDB()

    log.info("Setting up agents...")
    browser = BrowserAgent()
    agent_manager = AgentManager(memory=memory, browser=browser)

    log.info("Setting up chat engine...")
    chat_engine = ChatEngine(memory=memory, llm=llm, training_db=training_db)

    import asyncio
    reactive_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    chat_engine.set_learning_queue(reactive_queue)

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
    controller.reactive_queue = reactive_queue

    ws_server = WebSocketStateServer(controller)

    trainer.set_event_callback(ws_server.push_event)
    agent_manager.set_event_callback(ws_server.push_event)

    log.info("Initializing Temporal Pipeline...")
    pipeline = TemporalPipeline(
        memory=memory,
        batch_engine=batch_engine,
        hei=hei,
        event_callback=ws_server.push_event,
    )
    trainer.set_temporal_pipeline(pipeline)
    trainer.set_hei(hei)

    log.info("Setting up language trainer...")
    from lqnn.training.language_trainer import LanguageTrainer
    from ui.websocket_server import set_language_trainer
    language_trainer = LanguageTrainer(memory=memory)
    language_trainer.set_event_callback(ws_server.push_event)
    set_language_trainer(language_trainer)

    log.info("Setting up knowledge ingestion pipeline...")
    from lqnn.ingestion.processor import KnowledgeIngestionPipeline
    from lqnn.ingestion.rabbit_queue import RabbitIngestionQueue
    ingestion = KnowledgeIngestionPipeline(
        memory=memory,
        event_callback=ws_server.push_event,
    )
    ingestion.set_batch_engine(batch_engine)
    ingestion.set_hei(hei)
    controller.set_ingestion_pipeline(ingestion)
    rabbitmq_url = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    controller.ingestion_queue = RabbitIngestionQueue(
        rabbitmq_url=rabbitmq_url,
        ingestion_pipeline=ingestion,
        event_callback=ws_server.push_event,
    )

    app = create_app(controller=controller, ws_server=ws_server, trainer=trainer)

    log.info("System initialized with quantum processing revolution. Starting server...")
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

# LQNN v2 - Liquid Quantum Neural Network: Quantum Associative Brain

An AI system that learns like the human brain -- encoding concepts as associative
vectors in a shared visual-linguistic space, with autonomous web crawling,
volatile memory consolidation, and continuous self-evolution.

## Architecture

- **OpenCLIP ViT-B/32**: Encodes images and text into the same 512-d vector space
- **Phi-3.5-mini-instruct (4-bit)**: Text generation, association extraction, reasoning
- **ChromaDB**: Persistent vector storage with volatility metadata
- **MongoDB**: Training logs, chat history, agent activity
- **Autonomous Agents**: Web crawling, content analysis, quality judging

## How It Works

1. **See "banana"** -> CLIP encodes it into a vector
2. **LLM generates associations**: yellow, sweet, fruit, monkey, tropical...
3. **Each association becomes a CLIP vector** stored in ChromaDB
4. **All vectors coexist in superposition** with volatility scores
5. **On query**, the nearest vectors collapse to form an answer
6. **Consolidation cycles** crystallize stable knowledge, prune unused concepts
7. **Agents autonomously crawl** the web for images and text to learn from

## Hardware Requirements

- NVIDIA GPU with 8GB+ VRAM (tested on RTX 4060)
- 32GB RAM
- Docker with nvidia-container-toolkit

## Quick Start (Docker)

```bash
docker compose up -d --build
```

The brain starts learning automatically. Access the UI at:
- Chat: http://localhost:8000/chat
- Training Dashboard: http://localhost:8000/training

## Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main_loop.py
```

## Project Structure

```
lqnn/
  models/          # AI model wrappers (CLIP, Phi-3.5, downloader)
  core/            # Vector store (ChromaDB), associative memory
  agents/          # Web crawler, agent manager with judge
  training/        # Continuous training loop
  system/          # Chat engine, MongoDB logging
ui/
  app.py           # FastAPI application
  controls.py      # UI controller
  websocket_server.py
  static/          # Chat + Training dashboard HTML/CSS/JS
main_loop.py       # Entry point: starts everything
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /chat | Chat interface |
| GET | /training | Training dashboard |
| POST | /api/chat | Send chat message |
| POST | /api/search | Trigger web search |
| POST | /api/learn | Learn a concept |
| GET | /api/brain/status | Full brain status |
| GET | /api/training/status | Training metrics |
| POST | /api/training/cycle | Manual training cycle |
| GET | /api/agents/status | Agent status |
| POST | /api/agents/cycle | Manual agent cycle |
| POST | /api/consolidate | Run consolidation |
| POST | /api/self-play | Run self-play cycle |
| WS | /ws | WebSocket for real-time updates |

# LQNN v3 - Liquid Quantum Neural Network: Quantum Associative Brain

An AI system that learns like the human brain -- encoding concepts as associative
vectors in a shared visual-linguistic space, with autonomous web crawling,
volatile memory consolidation, and continuous self-evolution.

## Architecture

- **OpenCLIP ViT-B/32**: Encodes images and text into the same 512-d vector space
- **Qwen2.5-7B-Instruct (4-bit)**: Text generation, association extraction, reasoning
- **ChromaDB**: Persistent vector storage with volatility metadata
- **MongoDB**: Training logs, chat history, agent activity
- **Autonomous Agents**: Web crawling, content analysis, quality judging with CLIP validation

## How It Works

1. **See "banana"** -> CLIP encodes it into a vector (images weighted 70%, text 30%)
2. **LLM generates 30 associations** across categories: visual, sensory, semantic, relational
3. **Each association becomes a CLIP vector** stored in ChromaDB
4. **All vectors coexist in superposition** with volatility scores
5. **On query**, the nearest vectors collapse to form an answer
6. **Consolidation cycles** crystallize stable knowledge, prune unused concepts
7. **Network crystallization**: concepts with many interconnections stabilize faster
8. **Agents autonomously crawl** the web for images and text to learn from

## Training Phases

| Phase | Cycles | Focus |
|-------|--------|-------|
| **Visual Objects** | 1-100 | Concrete objects (banana, cat, car) with image priority |
| **Abstract Concepts** | 100+ | Science, history, emotions, derived from Phase 1 |
| **Self-Evolution** | 200+ concepts | AI decides what to learn from its own knowledge gaps |

## Hardware Requirements

- NVIDIA GPU with 8GB+ VRAM (tested on RTX 4060)
- 32GB RAM recommended
- Docker with nvidia-container-toolkit

## Quick Start (Docker)

```bash
docker compose up -d --build
```

The brain starts learning automatically. Access the terminal UI at:
- **http://localhost:8000/terminal**

## UI -- Hacker Terminal

The interface is a multi-panel terminal (htop/tmux style) with:
- **Chat Panel**: Interact with the quantum brain
- **Brain Status**: Real-time memory stats, model status
- **Training Log**: Live training cycle events
- **Agent Activity**: Web crawling, learning, judging events
- **Memory Map**: Concept list with volatility status (CRYSTAL/STABLE/VOLATILE)
- **System Metrics**: CPU, GPU, RAM usage in the status bar
- **Knowledge Base**: Upload PDFs, images, text files, and URLs as curated training data

## Knowledge Base

Access the Knowledge Base at **http://localhost:8000/knowledge** (or via the nav link in the terminal).

Supported data types:
- **PDF**: Papers, books, articles -- text extracted and images processed
- **Images**: JPG, PNG, WebP -- encoded with CLIP (70/30 visual/text weighting)
- **Text/Markdown**: Plain text files chunked semantically
- **URLs**: Articles and web pages fetched and processed

User-curated data receives special quantum treatment:
- **Lower initial volatility** (0.3 vs 1.0) -- curated knowledge starts more stable
- **Higher confidence** (0.7 vs 0.5) -- human-selected data is inherently more reliable
- **Curation bonus** during consolidation -- slower decay, harder to prune
- Concepts tagged as `user_curated` for differential treatment in the memory system

## Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main_loop.py
```

## Desktop Application (Tauri)

A native Linux desktop app wrapping the web UI for better performance and integration.

### Prerequisites

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Node.js (for Tauri CLI)
sudo apt install nodejs npm

# System libraries (Ubuntu/Debian)
sudo apt install libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf
```

### Build & Run

```bash
cd desktop
./build.sh
```

This produces `.deb` and `.AppImage` installers in `desktop/src-tauri/target/release/bundle/`.

To install the `.deb`:
```bash
sudo dpkg -i desktop/src-tauri/target/release/bundle/deb/*.deb
```

To run the AppImage directly:
```bash
chmod +x desktop/src-tauri/target/release/bundle/appimage/*.AppImage
./desktop/src-tauri/target/release/bundle/appimage/*.AppImage
```

**Note:** The desktop app connects to the LQNN backend (default `http://localhost:8000`). Make sure the Docker container is running first.

### Development

```bash
cd desktop
npm install
npm run dev
```

## Project Structure

```
lqnn/
  models/          # AI model wrappers (CLIP, Qwen2.5, downloader)
  core/            # Vector store (ChromaDB), associative memory
  agents/          # Web crawler, agent manager with judge
  training/        # Continuous training loop with phases
  system/          # Chat engine, MongoDB logging
  ingestion/       # Knowledge base pipeline (PDF, text, image, URL)
ui/
  app.py           # FastAPI application
  controls.py      # UI controller with system metrics
  websocket_server.py  # WebSocket with live event streaming
  static/          # Hacker terminal + knowledge base HTML/CSS/JS
desktop/
  src-tauri/       # Tauri (Rust) native desktop wrapper
  src/             # Loading page for desktop app
  build.sh         # Build script for .deb and .AppImage
main_loop.py       # Entry point: starts everything
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /terminal | Hacker terminal UI |
| POST | /api/chat | Send chat message |
| POST | /api/search | Trigger web search |
| POST | /api/learn | Learn a concept |
| GET | /api/brain/status | Full brain status |
| GET | /api/system/metrics | CPU/GPU/RAM metrics |
| GET | /api/training/status | Training metrics |
| POST | /api/training/start | Start continuous training |
| POST | /api/training/stop | Stop continuous training |
| POST | /api/training/cycle | Manual training cycle |
| GET | /api/agents/status | Agent status |
| POST | /api/agents/cycle | Manual agent cycle |
| POST | /api/consolidate | Run consolidation |
| POST | /api/self-play | Run self-play cycle |
| GET | /knowledge | Knowledge Base UI |
| POST | /api/knowledge/upload | Upload file (PDF, image, text) |
| POST | /api/knowledge/url | Ingest content from URL |
| WS | /ws | WebSocket for real-time updates |

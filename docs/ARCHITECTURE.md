# LQNN v2 - Architecture

## Overview

LQNN v2 is a **Quantum Associative Brain** -- an AI system that learns like the
human brain by encoding concepts as associative vectors in a shared
visual-linguistic space.

## Core Principle

> **"Structure is memory."**

Instead of storing knowledge as weight matrices in a neural network, LQNN stores
knowledge as **vectors in a shared embedding space** (CLIP) with **volatile
metadata** that controls how long each piece of knowledge persists.

## System Architecture

```
+-----------------------------------------------------------+
|                Docker Container (GPU)                       |
|                                                           |
|  +-----------+  +-----------+  +---------------------+   |
|  | OpenCLIP  |  | Phi-3.5   |  | ChromaDB            |   |
|  | ViT-B/32  |  | mini-inst |  | (Vector Store)      |   |
|  +-----------+  +-----------+  +---------------------+   |
|       |              |              ^          |          |
|       v              v              |          v          |
|  +---------------------------------------------+        |
|  |       Associative Memory (Brain)             |        |
|  |  - learn_concept()    - query()              |        |
|  |  - consolidate()      - self_play_cycle()    |        |
|  +---------------------------------------------+        |
|       ^              |              ^                    |
|       |              v              |                    |
|  +-----------+  +-----------+  +-----------+            |
|  | Browser   |  | Chat      |  | Continuous|            |
|  | Agent     |  | Engine    |  | Trainer   |            |
|  +-----------+  +-----------+  +-----------+            |
|       ^              |              |                    |
|       |              v              v                    |
|  +-----------+  +----------------------------+          |
|  | Agent     |  | FastAPI + WebSocket UI     |          |
|  | Manager   |  +----------------------------+          |
|  | + Judge   |       |                                  |
|  +-----------+       v                                  |
|                 +-----------+                            |
|                 | MongoDB   |                            |
|                 | (Logs)    |                            |
|                 +-----------+                            |
+-----------------------------------------------------------+
```

## Component Details

### AI Models (`lqnn/models/`)

| Component | File | Purpose |
|-----------|------|---------|
| Model Downloader | `downloader.py` | Auto-downloads OpenCLIP and Phi-3.5 on first run |
| CLIP Encoder | `clip_encoder.py` | Encodes images and text into 512-d shared vector space |
| LLM Engine | `llm_engine.py` | Text generation, association extraction, Q&A |

**OpenCLIP ViT-B/32** creates vectors where images and text that share meaning
are close together. This enables:
- "banana" (text) and a photo of a banana (image) produce similar vectors
- Similarity search finds related concepts across modalities

**Phi-3.5-mini-instruct (4-bit)** is quantized to fit in ~3GB VRAM:
- Generates association lists for new concepts
- Answers questions grounded on retrieved context
- Describes images for concept naming

### Core Brain (`lqnn/core/`)

| Component | File | Purpose |
|-----------|------|---------|
| Vector Store | `vector_store.py` | ChromaDB wrapper with volatility metadata |
| Associative Memory | `associative_memory.py` | The quantum brain logic |

The **Associative Memory** implements the learning cycle:

1. **Encode** -- CLIP encodes input (text/image) into a 512-d vector
2. **Associate** -- LLM generates related concepts, each encoded by CLIP
3. **Store** -- All vectors stored in ChromaDB with `volatility=1.0`
4. **Query (Collapse)** -- Find nearest vectors, build context, generate answer
5. **Consolidate** -- Frequently used vectors crystallize (low volatility);
   unused ones decay and are eventually pruned
6. **Self-play** -- Query own knowledge to find and fill gaps

### Agents (`lqnn/agents/`)

| Component | File | Purpose |
|-----------|------|---------|
| Browser Agent | `browser_agent.py` | Async web crawler (aiohttp + BeautifulSoup) |
| Agent Manager | `manager.py` | Orchestrates crawling, judging, learning |

The **Agent Manager** runs an autonomous knowledge acquisition pipeline:
1. Detect knowledge gaps (volatile or missing concepts)
2. Search the web via DuckDuckGo
3. Fetch pages, extract text and images
4. Judge quality (text coherence, image validity, deduplication)
5. Feed approved content into the associative memory

### Training (`lqnn/training/`)

| Component | File | Purpose |
|-----------|------|---------|
| Continuous Trainer | `continuous_trainer.py` | Never-stopping learning loop |

Runs as an asyncio background task:
- Agent cycle every 60 seconds
- Consolidation every 10 cycles
- Self-play every 3 cycles
- Metrics logged to MongoDB every 5 cycles

### System (`lqnn/system/`)

| Component | File | Purpose |
|-----------|------|---------|
| Chat Engine | `chat_engine.py` | CLIP retrieval + LLM generation |
| Training DB | `training_db.py` | MongoDB integration for logging |

### UI (`ui/`)

| Component | File | Purpose |
|-----------|------|---------|
| FastAPI App | `app.py` | REST API + static file serving |
| Controller | `controls.py` | Bridges UI to brain operations |
| WebSocket | `websocket_server.py` | Real-time state push |
| Chat UI | `static/chat.html` | Terminal-style chat interface |
| Training UI | `static/training.html` | Training metrics dashboard |

## Data Flow

### Learning Flow
```
Input (text/image)
  -> CLIP encode -> primary vector (512-d)
  -> LLM extract associations -> [word1, word2, ..., word20]
  -> CLIP encode each association -> [vec1, vec2, ..., vec20]
  -> Store all in ChromaDB with volatility=1.0
```

### Query Flow
```
Question (text)
  -> CLIP encode -> query vector (512-d)
  -> ChromaDB nearest-neighbor search -> matched concepts
  -> Retrieve associations for matched concepts
  -> Build context string from matches
  -> LLM answer with context -> response
  -> Update access_count for matched concepts
```

### Consolidation Flow
```
Every 10 training cycles:
  For each concept in ChromaDB:
    if access_count > 5:
      decrease volatility (crystallize)
    elif last_accessed > 24h ago:
      increase volatility
    if volatility > 0.95:
      DELETE (brain forgets)
```

## Memory Model

Inspired by human memory stages:

| Stage | Volatility | Behavior |
|-------|-----------|----------|
| **Short-term** | 0.8 - 1.0 | New knowledge, easily lost |
| **Medium-term** | 0.3 - 0.8 | Accessed a few times, semi-stable |
| **Long-term (Crystal)** | 0.0 - 0.3 | Frequently used, permanent |
| **Pruned** | > 0.95 after decay | Deleted by consolidation |

# LQNN v3 - Architecture

## Overview

LQNN v3 is a **Quantum Associative Brain** -- an AI system that learns like the
human brain by encoding concepts as associative vectors in a shared
visual-linguistic space with phased training and autonomous self-evolution.

## Core Principle

> **"Structure is memory."**

Instead of storing knowledge as weight matrices in a neural network, LQNN stores
knowledge as **vectors in a shared embedding space** (CLIP) with **volatile
metadata** that controls how long each piece of knowledge persists.

## System Architecture

```
+-------------------------------------------------------------------+
|                    Docker Container (GPU)                           |
|                                                                   |
|  +-----------+  +-----------+  +-------------------------+        |
|  | OpenCLIP  |  | Qwen2.5   |  | ChromaDB               |        |
|  | ViT-B/32  |  | 7B 4-bit  |  | (Vector Store)         |        |
|  +-----------+  +-----------+  +-------------------------+        |
|       |              |              ^          |                   |
|       v              v              |          v                   |
|  +-----------------------------------------------------+         |
|  |          Associative Memory (Quantum Brain)          |         |
|  |  - learn_concept()    (70% visual / 30% text)        |         |
|  |  - learn_multi_image()                               |         |
|  |  - query() (collapse)                                |         |
|  |  - consolidate() (network-aware crystallization)     |         |
|  |  - self_play_cycle()                                 |         |
|  +-----------------------------------------------------+         |
|       ^              |              ^                              |
|       |              v              |                              |
|  +-----------+  +-----------+  +-------------------+              |
|  | Browser   |  | Chat      |  | Continuous Trainer|              |
|  | Agent     |  | Engine    |  | (Phase-aware)     |              |
|  +-----------+  +-----------+  +-------------------+              |
|       ^              |         |  Phase 1: Visual   |              |
|       |              |         |  Phase 2: Abstract  |              |
|  +-----------+       |         |  Phase 3: Self-Evo  |              |
|  | Agent     |       v         +-------------------+              |
|  | Manager   |  +-------------------------------+                 |
|  | + Judge   |  | FastAPI + WebSocket           |                 |
|  | + CLIP    |  | Hacker Terminal UI            |                 |
|  +-----------+  +-------------------------------+                 |
|                      |                                             |
|                 +-----------+                                      |
|                 | MongoDB   |                                      |
|                 | (Logs)    |                                      |
|                 +-----------+                                      |
+-------------------------------------------------------------------+
```

## Component Details

### AI Models (`lqnn/models/`)

| Component | File | Purpose |
|-----------|------|---------|
| Model Downloader | `downloader.py` | Auto-downloads OpenCLIP and Qwen2.5-7B on first run |
| CLIP Encoder | `clip_encoder.py` | Encodes images and text into 512-d shared vector space |
| LLM Engine | `llm_engine.py` | Text generation, categorized association extraction, Q&A, relevance judging |

**OpenCLIP ViT-B/32** creates vectors where images and text that share meaning
are close together. This enables:
- "banana" (text) and a photo of a banana (image) produce similar vectors
- Similarity search finds related concepts across modalities

**Qwen2.5-7B-Instruct (4-bit)** is quantized to fit in ~4GB VRAM:
- Generates 30 categorized associations (visual, sensory, semantic, relational)
- Answers questions grounded on retrieved context with system prompts
- Judges text relevance to concepts with numerical scoring
- Excellent multilingual support (Portuguese, English, etc.)

### Core Brain (`lqnn/core/`)

| Component | File | Purpose |
|-----------|------|---------|
| Vector Store | `vector_store.py` | ChromaDB wrapper with volatility metadata |
| Associative Memory | `associative_memory.py` | The quantum brain logic |

The **Associative Memory** implements visual-first learning:

1. **Encode** -- CLIP encodes input (70% image / 30% text when image available)
2. **Multi-Image** -- Average multiple image vectors for robust representations
3. **Associate** -- LLM generates 30 categorized associations, each CLIP-encoded
4. **Store** -- All vectors stored in ChromaDB with `volatility=1.0`
5. **Query (Collapse)** -- Find nearest vectors, build context, generate answer
6. **Consolidate** -- Network-aware: concepts with many interconnections crystallize faster
7. **Self-play** -- Query own knowledge to find and fill gaps

### Agents (`lqnn/agents/`)

| Component | File | Purpose |
|-----------|------|---------|
| Browser Agent | `browser_agent.py` | Async web crawler (aiohttp + BeautifulSoup) |
| Agent Manager | `manager.py` | Phase-aware orchestration with CLIP-validated judging |
| Judge Agent | `manager.py` | Text coherence, image validity, CLIP relevance scoring |

The **Agent Manager** is phase-aware:
- **Visual phase**: Prioritizes image searches, validates image-concept relevance via CLIP
- **Abstract phase**: Uses association-derived seeds for expansion
- **Self-evolution**: Derives new topics from existing knowledge graph

### Training (`lqnn/training/`)

| Component | File | Purpose |
|-----------|------|---------|
| Continuous Trainer | `continuous_trainer.py` | Phase-aware training with event broadcasting |

Training phases:
- **Phase 1 (Visual Objects)**: 50 seed concepts (banana, cat, car...) with image priority
- **Phase 2 (Abstract Concepts)**: 35 abstract seeds (gravity, democracy, music...)
- **Phase 3 (Self-Evolution)**: AI derives new topics from its own associations

### System (`lqnn/system/`)

| Component | File | Purpose |
|-----------|------|---------|
| Chat Engine | `chat_engine.py` | CLIP retrieval + LLM generation with error handling |
| Training DB | `training_db.py` | MongoDB integration for logging |

### UI (`ui/`)

| Component | File | Purpose |
|-----------|------|---------|
| FastAPI App | `app.py` | REST API + static file serving + system metrics |
| Controller | `controls.py` | Bridges UI to brain with system metrics collection |
| WebSocket | `websocket_server.py` | Real-time state push + live event streaming |
| Terminal UI | `static/terminal.html` | Multi-panel hacker terminal interface |

## Data Flow

### Learning Flow (Visual-first)
```
Input (text + optional image)
  -> CLIP encode text -> text vector (512-d)
  -> CLIP encode image -> image vector (512-d)
  -> Weighted average (70% image + 30% text) -> primary vector
  -> LLM extract 30 categorized associations
  -> CLIP encode each association -> [vec1, ..., vec30]
  -> Store all in ChromaDB with volatility=1.0
```

### Query Flow (Collapse)
```
Question (text)
  -> CLIP encode -> query vector (512-d)
  -> ChromaDB nearest-neighbor search -> matched concepts
  -> Retrieve associations for matched concepts
  -> Build context string from matches
  -> LLM answer with context (system prompt) -> response
  -> Update access_count for matched concepts
```

### Consolidation Flow (Network-aware)
```
Every 10 training cycles:
  Count association links per concept (network bonus)
  For each concept in ChromaDB:
    if access_count > 5:
      decrease volatility + network bonus (crystallize faster)
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

Network crystallization bonus: concepts with 20+ associations crystallize
~0.2 volatility points faster per consolidation cycle.

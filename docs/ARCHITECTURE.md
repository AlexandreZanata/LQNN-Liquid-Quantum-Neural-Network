# LQNN v3.1 - Architecture

## Overview

LQNN v3.1 is a **Quantum Associative Brain** -- an AI system that learns like the
human brain by encoding concepts as associative vectors in a shared
visual-linguistic space with phased training, autonomous self-evolution,
and a 100x-throughput quantum processing engine.

## Core Principle

> **"Structure is memory."**

Instead of storing knowledge as weight matrices in a neural network, LQNN stores
knowledge as **vectors in a shared embedding space** (CLIP) with **volatile
metadata** that controls how long each piece of knowledge persists.

## Models

| Model | Parameters | Quantisation | VRAM | Purpose |
|-------|-----------|--------------|------|---------|
| **Qwen2.5-7B-Instruct** | 7.61 billion | NF4 4-bit | ~4.0 GB | Generation, association, reasoning |
| **OpenCLIP ViT-B/32** | 151 million | FP32 | ~0.5 GB | Multimodal 512-d encoding |

## System Architecture

```
+---------------------------------------------------------------------+
|                    Docker Container (GPU)                              |
|                                                                       |
|  +-----------+  +-----------+  +---------------------------+          |
|  | OpenCLIP  |  | Qwen2.5   |  | ChromaDB                  |          |
|  | ViT-B/32  |  | 7B 4-bit  |  | (Vector Store + f16)      |          |
|  | +CUDA str.|  | +batch    |  | +dirty-set +crystal tier  |          |
|  | +hot cache|  | assoc gen |  +---------------------------+          |
|  +-----------+  +-----------+           ^        |                    |
|       |              |                  |        v                    |
|  +----------------------------------------------------------+        |
|  |         Quantum Batch Engine (priority lanes)             |        |
|  |  URGENT (chat) -> instant | NORMAL (ingest) -> batched    |        |
|  +----------------------------------------------------------+        |
|       |              |              |         |                       |
|       v              v              v         v                       |
|  +----------------------------------------------------------+        |
|  |     Associative Memory v2 (Quantum Brain)                 |        |
|  |  - Wave-collapse (amplitudes, interference, multi-hop)    |        |
|  |  - Adaptive n_results (5 -> 25 -> 50)                     |        |
|  |  - Resonance detection (auto meta-concepts)               |        |
|  |  - 1-4 adaptive association workers                        |        |
|  |  - Crystallised tier (in-memory fast index)               |        |
|  +----------------------------------------------------------+        |
|       ^              |              ^         ^                       |
|       |              v              |         |                       |
|  +-----------+  +-----------+  +-----------------------+              |
|  | Browser   |  | Chat      |  | Temporal Pipeline     |              |
|  | Agent     |  | Engine    |  | Stage 1: Perception   |              |
|  +-----------+  | +multi-hop|  | Stage 2: Encoding     |              |
|       ^         +-----------+  | Stage 3: Integration  |              |
|       |              |         | Stage 4: Consolidation|              |
|  +-----------+       |         | Stage 5: Resonance    |              |
|  | Agent     |       v         +-----------------------+              |
|  | Manager   |  +-------------------------------+                     |
|  | + Judge   |  | FastAPI + WebSocket           |                     |
|  +-----------+  | Hacker Terminal UI            |                     |
|       |         +-------------------------------+                     |
|  +----------+        |                                                |
|  | HEI      |   +-----------+                                         |
|  | L0/L1/L2 |   | MongoDB   |                                         |
|  +----------+   +-----------+                                         |
+---------------------------------------------------------------------+
```

## Component Details

### AI Models (`lqnn/models/`)

| Component | File | Purpose |
|-----------|------|---------|
| Model Downloader | `downloader.py` | Auto-downloads OpenCLIP and Qwen2.5-7B on first run |
| CLIP Encoder | `clip_encoder.py` | 512-d encoding with GPU hot cache, pinned memory, dual CUDA streams |
| LLM Engine | `llm_engine.py` | Text generation, batch association extraction (5 concepts/call), Q&A |

### Core Brain (`lqnn/core/`)

| Component | File | Purpose |
|-----------|------|---------|
| Vector Store | `vector_store.py` | ChromaDB with f16 storage, batch ops, dirty-set, crystal tier |
| Associative Memory | `associative_memory.py` | Wave-collapse engine with multi-hop probing |
| Batch Engine | `quantum_batch_engine.py` | GPU-saturating batch pipelines with priority lanes |
| Entanglement Index | `entanglement_index.py` | 3-level hierarchical cluster index (HEI) |
| Temporal Pipeline | `temporal_pipeline.py` | 5-stage overlapping continuous processing |
| Phase Space | `phase_space.py` | Matryoshka search, adaptive chunks, diversity pruning |

### Data Flow

#### Learning Flow (Visual-first + Batch)
```
Input (text + optional image)
  -> Batch Engine (NORMAL lane)
  -> GPU batch CLIP encode (up to 64 items)
  -> Float16 storage in ChromaDB
  -> HEI incremental cluster assignment
  -> Background association workers (1-4 threads)
  -> Temporal Pipeline integration stage
```

#### Query Flow (Wave-Collapse)
```
Question (text)
  -> GPU hot cache check (skip encode if cached)
  -> CLIP encode via CUDA stream
  -> Crystal tier fast scan (in-memory numpy)
  -> Adaptive ANN search (n=5 -> 25 -> 50)
  -> Probability amplitude computation
  -> Interference patterns (constructive/destructive)
  -> Multi-hop probes (5 entangled concepts -> second-order knowledge)
  -> Build rich context (~15 000 chars, 30 fragments)
  -> LLM answer with expanded context (up to 500 tokens)
```

#### Consolidation Flow (Incremental)
```
Every consolidation interval:
  Pop dirty_set (only modified concept IDs)
  For each dirty concept:
    Compute volatility with network bonus + curation bonus
    if crystallised (vol < 0.2):
      Promote to in-memory crystal tier
    if pruned (vol > 0.95):
      Delete from Chroma + crystal tier
  Resonance detection (every 300s):
    Find concept pairs sharing >5 associations
    Create meta-concepts (emergent knowledge)
  HEI rebuild (if >1000 new concepts since last build)
```

## Memory Model

| Stage | Volatility | Location | Behaviour |
|-------|-----------|----------|-----------|
| **Short-term** | 0.8 - 1.0 | Chroma only | Fragile, easily pruned |
| **Medium-term** | 0.3 - 0.8 | Chroma | Semi-stable |
| **Long-term (Crystal)** | 0.0 - 0.3 | Chroma + in-memory tier | Sub-ms lookups, permanent |
| **Pruned** | > 0.95 | Deleted | Brain forgets |

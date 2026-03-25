# LQNN v5.0 - Quantum Velocity Architecture

## Overview

LQNN v5.0 is a **Quantum Associative Brain** -- a self-evolving AI organism that
learns like the human brain by encoding concepts as associative vectors in a shared
visual-linguistic space with Flash Attention 2 / SDPA acceleration,
adaptive priority-lane batching, **GPU Quantum Exclusion** (100% GPU for chat),
a **Quantum Tunneling Fast Path** for low-latency inference, a **Quantum
Decoherence Shield** that filters corrupted data, and a living feedback loop
that strengthens knowledge through response resonance.

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
| Associative Memory | `associative_memory.py` | Wave-collapse engine with HEI pre-filter, batch engine, superposition context assembly |
| Batch Engine | `quantum_batch_engine.py` | GPU-saturating batch pipelines with priority sorting + batched queries |
| Entanglement Index | `entanglement_index.py` | 3-level hierarchical cluster index (HEI), wired into query path via narrow() |
| Temporal Pipeline | `temporal_pipeline.py` | 5-stage overlapping continuous processing, batch engine integrated |
| Phase Space | `phase_space.py` | Matryoshka search, adaptive chunks, diversity pruning |

### Data Flow

#### Learning Flow (Library-focused + Batch)
```
Knowledge Library (PDF, text, images via UI upload)
  -> Ingestion Pipeline (extract, chunk, validate)
  -> Batch Engine (NORMAL lane)
  -> GPU batch CLIP encode (up to 64 items)
  -> Float16 storage in ChromaDB
  -> HEI incremental cluster assignment
  -> Background association workers (1-8 threads, GPU-priority aware)
  -> Temporal Pipeline integration stage

Training Loop (every 60s, NO web crawling):
  -> Consolidation (every 2 cycles: crystallise / prune)
  -> Self-play (every 2 cycles: validate own knowledge)
  -> Library reinforcement (every 3 cycles: strengthen sparse associations)
  -> HEI rebuild (every 30 cycles)

Agents (ON-DEMAND ONLY, not in training loop):
  -> Activated only when chat confidence < 0.08 (reactive search)
  -> Or when user explicitly triggers search from UI
  -> Extreme quality filtering: decoherence shield + boilerplate + CLIP gate
```

#### Query Flow (Wave-Collapse v5 + Quantum Tunneling)
```
Question (text)
  -> GPU Exclusion Gate ACTIVATE (pause all background LLM work)
  -> GPU hot cache check (4000 vectors, skip encode if cached)
  -> CLIP encode via Batch Engine (URGENT lane, instant wake)
  -> HEI.narrow() + Crystal tier scan (parallel, 8 workers)
  -> Adaptive ANN search with ID filter (n=5 -> 25 -> 50)
  -> Probability amplitude computation
  -> Interference patterns (constructive/destructive)
  -> Dynamic multi-hop probes (3-12 based on confidence history)
  -> Superposition Context Assembly:
      Quantum Decoherence Shield (filter garbage/corrupted data)
      Dedup + relevance ranking + hierarchical priority
      Crystal > Primary > Multi-hop > Associations
      Budget: ~8000 chars (~2K tokens, Heisenberg-optimal for 7B)
  -> Conversation history injection (last 10 turns)
  -> Quantum Tunneling Fast Path (most queries):
      Single-pass 1500 tokens with repetition_penalty=1.15
  -> Complex queries only (rare):
      Phase 1: Streamed outline (150 tokens, 3-5 sections)
      Phase 2: Section generation (up to 5 x 800 tokens)
      (No verification in streaming -- tunnels directly to answer)
      Total output: up to 6000 tokens
  -> Response quality feedback (volatility reinforcement)
  -> Neural resonance amplification (association strengthening)
  -> GPU Exclusion Gate RELEASE (resume background work)
  -> Cancellable at any phase via StoppingCriteria
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

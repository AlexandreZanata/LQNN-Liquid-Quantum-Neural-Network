# LQNN v3 - API Reference

## REST Endpoints

### Pages

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Redirect to `/terminal` |
| `GET` | `/terminal` | Hacker terminal interface (HTML) |
| `GET` | `/chat` | Redirect to `/terminal` |
| `GET` | `/training` | Redirect to `/terminal` |

### Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns `{"status": "ready|loading|alive", "version": "3.0.0"}` |

### System Metrics

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/system/metrics` | CPU, GPU, RAM usage |

### Chat

| Method | Path | Body | Response |
|--------|------|------|----------|
| `POST` | `/api/chat` | `{"text": "question"}` | `{"response": "...", "confidence": 0.8, "concepts": [...], "duration_ms": 150, "status": "ok"}` |

### Knowledge

| Method | Path | Body | Response |
|--------|------|------|----------|
| `POST` | `/api/learn` | `{"concept": "banana"}` | `{"concept": "banana", "associations": 30, "volatility": 1.0}` |
| `POST` | `/api/search` | `{"query": "quantum physics"}` | `{"cycle": 1, "concepts_learned": 5, "images_processed": 2, "duration_s": 3.5}` |
| `POST` | `/api/consolidate` | -- | `{"pruned": 2, "crystallized": 5, "decayed": 3}` |
| `POST` | `/api/self-play` | -- | `{"action": "validated", "concept": "banana", "confidence": 0.85}` |

### Brain Status

| Method | Path | Response |
|--------|------|----------|
| `GET` | `/api/brain/status` | Full brain state snapshot (memory, training, agents, concepts, logs, system) |
| `GET` | `/api/memory/stats` | `{"concepts": 150, "associations": 3000, "learn_count": 45, "query_count": 20}` |

### Training

| Method | Path | Response |
|--------|------|----------|
| `GET` | `/api/training/status` | Training status (running, cycle, phase, uptime, latest_metrics) |
| `POST` | `/api/training/start` | Start continuous training |
| `POST` | `/api/training/stop` | Stop continuous training |
| `POST` | `/api/training/cycle` | Trigger one manual training cycle |

### Agents

| Method | Path | Response |
|--------|------|----------|
| `GET` | `/api/agents/status` | Agent status (cycle, online, phase, gap_queue_size) |
| `POST` | `/api/agents/cycle` | Trigger one manual agent cycle |

## WebSocket

### Endpoint

```
ws://localhost:8000/ws
```

### Client -> Server Messages

All messages are JSON with an `action` field:

```json
{"action": "chat", "text": "what is a banana?"}
{"action": "search", "query": "quantum physics"}
{"action": "learn", "concept": "banana"}
{"action": "consolidate"}
{"action": "self_play"}
{"action": "train_cycle"}
{"action": "agent_cycle"}
{"action": "start_training"}
{"action": "stop_training"}
{"action": "get_state"}
```

### Server -> Client Messages

State broadcasts (every 0.5s):
```json
{
  "type": "state",
  "memory": {"concepts": 150, "associations": 3000, ...},
  "training": {"running": true, "cycle": 42, "phase": "visual_objects", ...},
  "agents": {"cycle": 15, "online": true, "phase": "visual_objects", ...},
  "chat_history": [...],
  "recent_concepts": [{"concept": "banana", "volatility": 0.1, "status": "CRYSTAL", ...}],
  "training_log": [{"type": "cycle_end", "cycle": 42, ...}],
  "agent_activity": [{"type": "learn", "concept": "banana", ...}],
  "system": {"cpu_percent": 45, "gpu_used_gb": 3.2, ...}
}
```

Live events (pushed instantly):
```json
{
  "type": "live_event",
  "type": "learn",
  "concept": "banana",
  "associations": 30,
  "timestamp": 1711324800
}
```

Chat responses:
```json
{
  "type": "chat_response",
  "response": "A banana is a tropical fruit...",
  "confidence": 0.85,
  "concepts": ["banana", "fruit", "tropical"],
  "duration_ms": 150,
  "status": "ok"
}
```

Error messages:
```json
{"type": "error", "message": "description"}
```

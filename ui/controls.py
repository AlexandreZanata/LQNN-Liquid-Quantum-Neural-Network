"""UI controller -- bridges the web interface to the brain system."""

from __future__ import annotations

import logging
from typing import Any
import subprocess
import time

from lqnn.core.associative_memory import AssociativeMemory
from lqnn.ingestion.processor import KnowledgeIngestionPipeline
from lqnn.system.chat_engine import ChatEngine
from lqnn.training.continuous_trainer import ContinuousTrainer
from lqnn.agents.manager import AgentManager
from ui.renderer import build_brain_payload

log = logging.getLogger(__name__)

_GPU_METRICS_CACHE_TTL_S = 2.0
_GPU_METRICS_LAST_TS = 0.0
_GPU_METRICS_LAST: dict[str, Any] = {
    "gpu_used_gb": 0.0,
    "gpu_total_gb": 0.0,
    "gpu_name": "N/A",
    "gpu_metric_source": "none",
    "gpu_metric_error": "",
}


def _read_gpu_metrics() -> dict[str, Any]:
    """Read GPU metrics with resilient fallback and short-lived cache."""
    global _GPU_METRICS_LAST_TS, _GPU_METRICS_LAST
    now = time.time()
    if now - _GPU_METRICS_LAST_TS < _GPU_METRICS_CACHE_TTL_S:
        return dict(_GPU_METRICS_LAST)

    metrics = dict(_GPU_METRICS_LAST)
    metrics["gpu_metric_error"] = ""

    try:
        import torch
    except Exception as exc:
        metrics.update({
            "gpu_used_gb": 0.0,
            "gpu_total_gb": 0.0,
            "gpu_name": "N/A",
            "gpu_metric_source": "none",
            "gpu_metric_error": f"torch_import_failed:{exc}",
        })
        _GPU_METRICS_LAST = metrics
        _GPU_METRICS_LAST_TS = now
        return metrics

    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        metrics.update({
            "gpu_used_gb": 0.0,
            "gpu_total_gb": 0.0,
            "gpu_name": "N/A",
            "gpu_metric_source": "no_cuda",
            "gpu_metric_error": "",
        })
        _GPU_METRICS_LAST = metrics
        _GPU_METRICS_LAST_TS = now
        return metrics

    # Keep device name when CUDA is available, even if metric calls fail later.
    try:
        metrics["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        metrics["gpu_name"] = _GPU_METRICS_LAST.get("gpu_name", "N/A")

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip().splitlines()
        if out:
            used_mb, total_mb = [x.strip() for x in out[0].split(",")[:2]]
            metrics["gpu_used_gb"] = round(float(used_mb) / 1024, 1)
            metrics["gpu_total_gb"] = round(float(total_mb) / 1024, 1)
            metrics["gpu_metric_source"] = "nvidia_smi"
            metrics["gpu_metric_error"] = ""
            _GPU_METRICS_LAST = metrics
            _GPU_METRICS_LAST_TS = now
            return metrics
    except Exception as exc:
        metrics["gpu_metric_error"] = f"nvidia_smi_failed:{exc}"

    # Fallback: process-local CUDA memory.
    try:
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        metrics["gpu_used_gb"] = round(allocated, 1)
        metrics["gpu_total_gb"] = round(total, 1)
        metrics["gpu_metric_source"] = "torch_cuda"
        _GPU_METRICS_LAST = metrics
        _GPU_METRICS_LAST_TS = now
        return metrics
    except Exception as exc:
        # Last-known-good fallback avoids false 0/0 spikes.
        metrics["gpu_metric_source"] = "last_known_good"
        metrics["gpu_metric_error"] = (
            f"{metrics.get('gpu_metric_error', '')};torch_cuda_failed:{exc}"
        ).strip(";")
        _GPU_METRICS_LAST_TS = now
        return dict(_GPU_METRICS_LAST)


def _get_system_metrics(model_runtime_state: str = "unknown") -> dict[str, Any]:
    """Collect CPU, GPU, and memory metrics."""
    metrics: dict[str, Any] = {}
    try:
        import psutil
        metrics["cpu_percent"] = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()
        metrics["ram_used_gb"] = round(mem.used / (1024 ** 3), 1)
        metrics["ram_total_gb"] = round(mem.total / (1024 ** 3), 1)
    except ImportError:
        metrics["cpu_percent"] = 0
        metrics["ram_used_gb"] = 0
        metrics["ram_total_gb"] = 0

    gpu = _read_gpu_metrics()
    metrics.update(gpu)
    metrics["model_runtime_state"] = model_runtime_state
    metrics["cuda_available"] = gpu.get("gpu_metric_source") not in {"none", "no_cuda"}

    return metrics


class UIController:
    """Maps UI actions to brain operations."""

    def __init__(
        self,
        memory: AssociativeMemory,
        chat_engine: ChatEngine,
        trainer: ContinuousTrainer,
        agent_manager: AgentManager,
    ) -> None:
        self.memory = memory
        self.chat_engine = chat_engine
        self.trainer = trainer
        self.agent_manager = agent_manager
        self.ingestion: KnowledgeIngestionPipeline | None = None
        self.ingestion_queue = None

    def set_ingestion_pipeline(self, pipeline: KnowledgeIngestionPipeline) -> None:
        self.ingestion = pipeline

    def chat_turn(self, text: str) -> dict[str, Any]:
        return self.chat_engine.chat(text)

    async def search(self, query: str) -> dict[str, Any]:
        report = await self.agent_manager.request_search(query)
        return {
            "cycle": report.cycle,
            "concepts_learned": report.concepts_learned,
            "images_processed": report.images_processed,
            "duration_s": round(report.duration_s, 2),
        }

    async def run_agent_cycle(self) -> dict[str, Any]:
        report = await self.agent_manager.run_cycle()
        return {
            "cycle": report.cycle,
            "concepts_learned": report.concepts_learned,
            "images_processed": report.images_processed,
            "duration_s": round(report.duration_s, 2),
        }

    def consolidate(self) -> dict[str, Any]:
        return self.memory.consolidate()

    def self_play(self) -> dict[str, Any]:
        return self.memory.self_play_cycle()

    def learn_concept(self, concept: str) -> dict[str, Any]:
        state = self.memory.learn_concept(concept, source="ui_manual")
        return {
            "concept": state.concept,
            "associations": len(state.associations),
            "volatility": state.volatility,
        }

    async def train_cycle(self) -> dict[str, Any]:
        await self.trainer.run_manual_cycle()
        return self.trainer.latest_metrics() or {}

    def snapshot(self) -> dict[str, Any]:
        stable = self.memory.store.get_stable_concepts(threshold=0.3)
        volatile = self.memory.store.get_volatile_concepts(threshold=0.7)

        recent_concepts = []
        seen = set()
        for s in (stable + volatile)[:30]:
            doc = s.get("document", "")
            if not doc or doc in seen:
                continue
            seen.add(doc)
            meta = s.get("metadata", {})
            vol = meta.get("volatility", 1.0)
            if vol <= 0.2:
                status = "CRYSTAL"
            elif vol <= 0.5:
                status = "STABLE"
            else:
                status = "VOLATILE"
            recent_concepts.append({
                "concept": doc,
                "volatility": vol,
                "confidence": meta.get("confidence", 0.5),
                "access_count": meta.get("access_count", 0),
                "status": status,
            })

        recent_concepts.sort(key=lambda c: c["volatility"])

        if self.memory.llm.ready:
            model_runtime_state = "ready"
        elif self.memory.llm.loading:
            model_runtime_state = "loading"
        else:
            model_runtime_state = "idle"

        return build_brain_payload(
            memory_stats=self.memory.stats(),
            training_status=self.trainer.status(),
            agent_status=self.agent_manager.stats(),
            chat_history=self.chat_engine.chat_history[-20:],
            recent_concepts=recent_concepts[:20],
            training_log=self.trainer.training_log[-50:],
            agent_activity=self.agent_manager.activity_log[-50:],
            system_metrics=_get_system_metrics(model_runtime_state),
        )

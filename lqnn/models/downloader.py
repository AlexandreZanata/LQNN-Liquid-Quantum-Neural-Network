"""Auto-download OpenCLIP and Qwen2.5-7B on first run."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

CACHE_DIR = Path(os.environ.get("HF_HOME", "models/cache"))

QWEN_CACHE_DIR = CACHE_DIR / "models--Qwen--Qwen2.5-7B-Instruct"


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ensure_clip_model() -> tuple:
    """Download and return (model, preprocess, tokenizer) for OpenCLIP."""
    import open_clip

    log.info("Loading OpenCLIP %s (%s) ...", CLIP_MODEL_NAME, CLIP_PRETRAINED)
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
        cache_dir=str(CACHE_DIR),
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    device = _device()
    model = model.to(device).eval()
    log.info("OpenCLIP ready on %s", device)
    return model, preprocess, tokenizer


def ensure_llm_model() -> tuple:
    """Download and return (model, tokenizer) for Qwen2.5-7B with 4-bit quant."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    log.info("Loading %s (4-bit) ...", LLM_MODEL_ID)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_ID,
        cache_dir=str(CACHE_DIR),
        trust_remote_code=True,
    )

    device = _device()
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=bnb_config if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        cache_dir=str(CACHE_DIR),
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)

    log.info("LLM ready on %s (4-bit=%s)", device, device == "cuda")
    return model, tokenizer


def llm_cache_status() -> dict[str, Any]:
    """Return a quick local-cache status for Qwen2.5 files."""
    snapshots_root = QWEN_CACHE_DIR / "snapshots"
    if not snapshots_root.exists():
        return {
            "cached": False,
            "reason": "no_cache_dir",
            "snapshot": "",
            "size_gb": 0.0,
            "shards": 0,
        }

    snapshots = [p for p in snapshots_root.iterdir() if p.is_dir()]
    if not snapshots:
        return {
            "cached": False,
            "reason": "no_snapshots",
            "snapshot": "",
            "size_gb": 0.0,
            "shards": 0,
        }

    latest = max(snapshots, key=lambda p: p.stat().st_mtime)
    shard_files = list(latest.glob("*.safetensors"))
    has_index = (latest / "model.safetensors.index.json").exists()
    has_core = (
        (latest / "config.json").exists()
        and (latest / "tokenizer_config.json").exists()
    )
    total_bytes = sum(f.stat().st_size for f in latest.rglob("*") if f.is_file())
    size_gb = round(total_bytes / (1024 ** 3), 2)
    cached = has_core and (has_index or len(shard_files) > 0)

    return {
        "cached": cached,
        "reason": "ok" if cached else "partial_files",
        "snapshot": latest.name,
        "size_gb": size_gb,
        "shards": len(shard_files),
    }

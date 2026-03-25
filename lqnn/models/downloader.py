"""Auto-download OpenCLIP and Phi-3.5-mini on first run."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

log = logging.getLogger(__name__)

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

LLM_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

CACHE_DIR = Path(os.environ.get("HF_HOME", "models/cache"))


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
    """Download and return (model, tokenizer) for Phi-3.5-mini with 4-bit quant."""
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

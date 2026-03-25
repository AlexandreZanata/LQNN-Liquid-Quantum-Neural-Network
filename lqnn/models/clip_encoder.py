"""OpenCLIP wrapper with zero-copy streaming optimisations.

v2 additions:
- Pinned (page-locked) CUDA host memory for ~2x faster CPU<->GPU transfers
- Dual CUDA streams: one for CLIP compute, one for data movement
- GPU-resident hot-vector cache: 1000 most recent embeddings stay on CUDA
- encode_texts() uses true batched inference for maximum GPU saturation
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

_EMBED_DIM = 512  # ViT-B/32 output dimension
_HOT_CACHE_SIZE = 4000
_PINNED_BATCH_SLOTS = 128  # pre-allocated pinned memory slots


class _GPUHotCache:
    """LRU cache of CLIP vectors that live on the GPU.

    Avoids the CPU->GPU->CPU roundtrip for recently-seen texts.
    """

    def __init__(self, capacity: int = _HOT_CACHE_SIZE,
                 device: str = "cpu") -> None:
        self._capacity = capacity
        self._device = device
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> torch.Tensor | None:
        with self._lock:
            t = self._cache.get(key)
            if t is not None:
                self._cache.move_to_end(key)
            return t

    def put(self, key: str, tensor: torch.Tensor) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return
            if self._device != "cpu":
                tensor = tensor.to(self._device, non_blocking=True)
            self._cache[key] = tensor
            if len(self._cache) > self._capacity:
                _, evicted = self._cache.popitem(last=False)
                del evicted


class CLIPEncoder:
    """Encodes images and text into the same 512-d vector space via OpenCLIP.

    v2: pinned memory, dual CUDA streams, GPU hot cache.
    """

    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device: str = "cpu"
        self._ready = False
        self._hot_cache: _GPUHotCache | None = None

        # CUDA streams (created lazily after load)
        self._compute_stream: torch.cuda.Stream | None = None
        self._transfer_stream: torch.cuda.Stream | None = None

        # Pinned memory buffer for batch text encoding
        self._pinned_buffer: torch.Tensor | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def embed_dim(self) -> int:
        return _EMBED_DIM

    def load(self) -> None:
        from lqnn.models.downloader import ensure_clip_model, _device

        self._model, self._preprocess, self._tokenizer = ensure_clip_model()
        self._device = _device()
        self._ready = True

        self._hot_cache = _GPUHotCache(
            capacity=_HOT_CACHE_SIZE, device=self._device)

        if self._device == "cuda":
            try:
                self._compute_stream = torch.cuda.Stream()
                self._transfer_stream = torch.cuda.Stream()
                self._pinned_buffer = torch.empty(
                    (_PINNED_BATCH_SLOTS, _EMBED_DIM),
                    dtype=torch.float32,
                    pin_memory=True,
                )
                log.info("CLIP: pinned memory + dual CUDA streams ready")
            except Exception as exc:
                log.debug("CLIP: CUDA stream setup skipped: %s", exc)

    @torch.no_grad()
    def encode_image(self, image: Image.Image | bytes | str | Path) -> np.ndarray:
        """Encode a single image into a 512-d unit vector."""
        if not self._ready:
            self.load()

        if isinstance(image, (str,)):
            from pathlib import Path as P
            image = Image.open(P(image)).convert("RGB")
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image)).convert("RGB")
        elif hasattr(image, "__fspath__"):
            image = Image.open(image).convert("RGB")

        tensor = self._preprocess(image).unsqueeze(0).to(self._device)

        if self._compute_stream:
            with torch.cuda.stream(self._compute_stream):
                feat = self._model.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            self._compute_stream.synchronize()
        else:
            feat = self._model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        return feat.cpu().numpy().flatten().astype(np.float32)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text string into a 512-d unit vector.

        Checks the GPU hot cache first; on miss, encodes and caches.
        """
        if not self._ready:
            self.load()

        cache_key = text[:512]
        if self._hot_cache:
            cached = self._hot_cache.get(cache_key)
            if cached is not None:
                return cached.cpu().numpy().flatten().astype(np.float32)

        tokens = self._tokenizer([text]).to(self._device)

        if self._compute_stream:
            with torch.cuda.stream(self._compute_stream):
                feat = self._model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            self._compute_stream.synchronize()
        else:
            feat = self._model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        if self._hot_cache:
            self._hot_cache.put(cache_key, feat.squeeze(0))

        return feat.cpu().numpy().flatten().astype(np.float32)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Batch-encode multiple texts. Returns (N, 512) array.

        Uses pinned memory and CUDA streams for maximum throughput.
        """
        if not self._ready:
            self.load()

        if not texts:
            return np.empty((0, _EMBED_DIM), dtype=np.float32)

        results = np.empty((len(texts), _EMBED_DIM), dtype=np.float32)
        uncached_indices = []
        uncached_texts = []

        if self._hot_cache:
            for i, t in enumerate(texts):
                cached = self._hot_cache.get(t[:512])
                if cached is not None:
                    results[i] = cached.cpu().numpy().flatten()
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(t)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        if not uncached_texts:
            return results

        use_pinned = (self._pinned_buffer is not None
                      and self._transfer_stream is not None
                      and self._compute_stream is not None)

        tokens = self._tokenizer(uncached_texts).to(self._device)

        if self._compute_stream:
            with torch.cuda.stream(self._compute_stream):
                feats = self._model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            self._compute_stream.synchronize()
        else:
            feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        n_feats = feats.shape[0]
        if use_pinned and n_feats <= self._pinned_buffer.shape[0]:
            with torch.cuda.stream(self._transfer_stream):
                self._pinned_buffer[:n_feats].copy_(feats, non_blocking=True)
            self._transfer_stream.synchronize()
            feats_np = self._pinned_buffer[:n_feats].numpy().copy()
        else:
            feats_np = feats.cpu().numpy().astype(np.float32)

        if self._hot_cache:
            for idx, orig_idx in enumerate(uncached_indices):
                results[orig_idx] = feats_np[idx]
                self._hot_cache.put(texts[orig_idx][:512], feats[idx])
        else:
            for idx, orig_idx in enumerate(uncached_indices):
                results[orig_idx] = feats_np[idx]

        return results

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two vectors (both assumed unit-norm)."""
        return float(np.dot(vec_a, vec_b))

    def hot_cache_stats(self) -> dict:
        if not self._hot_cache:
            return {"size": 0, "capacity": 0}
        return {
            "size": len(self._hot_cache._cache),
            "capacity": self._hot_cache._capacity,
            "device": self._device,
            "cuda_streams": self._compute_stream is not None,
            "pinned_memory": self._pinned_buffer is not None,
        }

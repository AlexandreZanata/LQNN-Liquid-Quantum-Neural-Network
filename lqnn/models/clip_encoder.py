"""OpenCLIP wrapper for encoding images and text into a shared vector space."""

from __future__ import annotations

import logging
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

_EMBED_DIM = 512  # ViT-B/32 output dimension


class CLIPEncoder:
    """Encodes images and text into the same 512-d vector space via OpenCLIP."""

    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device: str = "cpu"
        self._ready = False

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
        feat = self._model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().flatten().astype(np.float32)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text string into a 512-d unit vector."""
        if not self._ready:
            self.load()

        tokens = self._tokenizer([text]).to(self._device)
        feat = self._model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().flatten().astype(np.float32)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Batch-encode multiple texts. Returns (N, 512) array."""
        if not self._ready:
            self.load()

        tokens = self._tokenizer(texts).to(self._device)
        feats = self._model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two vectors (both assumed unit-norm)."""
        return float(np.dot(vec_a, vec_b))

# Learning principle: one-shot concept acquisition stores class memory as
# activation topology snapshots, avoiding gradient-based training.

from __future__ import annotations

from typing import Tuple

import numpy as np

from lqnn.core.network import LiquidNetwork


class OneShotLearner:
    """Learns class concepts from a single example via topology memory."""

    def __init__(self) -> None:
        self.net = LiquidNetwork(initial_neurons=15, max_neurons=300)
        self.class_snapshots: dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        """Encode raw text into a compact local numeric signal."""
        chars = [ord(c) / 255.0 for c in text[:20]]
        chars += [0.0] * (20 - len(chars))

        vowels = sum(1 for c in text.lower() if c in "aeiou") / max(len(text), 1)
        features = np.array(
            chars
            + [
                len(text) / 100.0,
                vowels,
                text.count(" ") / max(len(text), 1),
            ],
            dtype=float,
        )
        return features

    def learn(self, label: str, example: str) -> None:
        """Memorize one class concept from a single text example."""
        print(f"\n[APRENDENDO] '{label}' a partir de: '{example}'")
        signal = self.encode(example)

        for _ in range(3):
            self.net.forward(signal)

        snapshot = self.net.forward(signal)
        self.class_snapshots[label] = snapshot.copy()
        print(f"  Padrao memorizado para '{label}': {int(snapshot.sum())} neuronios ativos")
        print(f"  {self.net.status()}")

    def classify(self, text: str) -> Tuple[str, float]:
        """Classify input text by similarity with stored topology snapshots."""
        if not self.class_snapshots:
            return "sem classes", 0.0

        signal = self.encode(text)
        current = self.net.forward(signal)

        best_label = "sem classes"
        best_score = -1.0

        for label, snapshot in self.class_snapshots.items():
            min_len = min(len(current), len(snapshot))
            if min_len == 0:
                continue

            a = current[:min_len]
            b = snapshot[:min_len]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a > 0.0 and norm_b > 0.0:
                score = float(np.dot(a, b) / (norm_a * norm_b))
            else:
                score = 0.0

            if score > best_score:
                best_score = score
                best_label = label

        return best_label, max(best_score, 0.0)

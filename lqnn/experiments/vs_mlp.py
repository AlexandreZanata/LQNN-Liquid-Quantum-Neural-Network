# Scientific principle: benchmark structural one-shot learning against a
# fixed-topology baseline for fair comparison.

from __future__ import annotations

import numpy as np

from lqnn.learning.one_shot import OneShotLearner


class FixedCentroidBaseline:
    """Simple fixed-topology baseline for one-shot comparison."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.centroids: dict[str, np.ndarray] = {}

    def learn(self, label: str, text: str) -> None:
        self.centroids[label] = self.encoder(text)

    def classify(self, text: str) -> tuple[str, float]:
        x = self.encoder(text)
        best_label = "sem classes"
        best_score = -1.0
        for label, centroid in self.centroids.items():
            denom = np.linalg.norm(x) * np.linalg.norm(centroid)
            score = float(np.dot(x, centroid) / denom) if denom > 0 else 0.0
            if score > best_score:
                best_score = score
                best_label = label
        return best_label, max(best_score, 0.0)


def run() -> dict[str, float]:
    """Compare LQNN one-shot against a fixed centroid baseline."""
    dataset = [
        ("positivo", "que produto maravilhoso"),
        ("negativo", "produto horrivel e ruim"),
        ("positivo", "gostei bastante, excelente"),
        ("negativo", "nao presta, terrivel"),
        ("positivo", "muito bom, recomendo"),
        ("negativo", "pessimo, nunca mais"),
    ]

    lqnn = OneShotLearner()
    baseline = FixedCentroidBaseline(lqnn.encode)

    lqnn.learn("positivo", dataset[0][1])
    lqnn.learn("negativo", dataset[1][1])
    baseline.learn("positivo", dataset[0][1])
    baseline.learn("negativo", dataset[1][1])

    eval_set = dataset[2:]
    lqnn_ok = 0
    base_ok = 0
    for expected, text in eval_set:
        lqnn_pred, _ = lqnn.classify(text)
        base_pred, _ = baseline.classify(text)
        lqnn_ok += int(lqnn_pred == expected)
        base_ok += int(base_pred == expected)

    result = {
        "lqnn_accuracy": lqnn_ok / len(eval_set),
        "baseline_accuracy": base_ok / len(eval_set),
    }

    print("=== LQNN vs baseline fixo ===")
    print(f"LQNN: {result['lqnn_accuracy'] * 100:.0f}%")
    print(f"Baseline fixo: {result['baseline_accuracy'] * 100:.0f}%")
    return result


if __name__ == "__main__":
    run()

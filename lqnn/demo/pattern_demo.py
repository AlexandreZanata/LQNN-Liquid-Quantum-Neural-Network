# Biological principle: repeated sensory patterns carve stable activation
# pathways in a self-organizing network.

from __future__ import annotations

import numpy as np

from lqnn.core.network import LiquidNetwork


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0.0
    av = a[:min_len]
    bv = b[:min_len]
    na = np.linalg.norm(av)
    nb = np.linalg.norm(bv)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def run() -> None:
    print("=== LQNN - Pattern Demo (one-shot) ===")
    net = LiquidNetwork(initial_neurons=18, max_neurons=280)

    cross = np.array([
        1, 0, 1,
        0, 1, 0,
        1, 0, 1,
    ], dtype=float)
    line = np.array([
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
    ], dtype=float)

    for _ in range(4):
        net.forward(cross)
    snap_cross = net.forward(cross)

    for _ in range(4):
        net.forward(line)
    snap_line = net.forward(line)

    tests = [
        ("cross", cross),
        ("line", line),
        ("cross_noisy", np.clip(cross + np.random.normal(0.0, 0.1, len(cross)), 0.0, 1.0)),
        ("line_noisy", np.clip(line + np.random.normal(0.0, 0.1, len(line)), 0.0, 1.0)),
    ]

    print("\n--- Classificando padroes ---")
    for name, signal in tests:
        current = net.forward(signal)
        score_cross = _cosine(current, snap_cross)
        score_line = _cosine(current, snap_line)
        label = "cross" if score_cross >= score_line else "line"
        print(
            f"  {name:10s} -> {label:5s} "
            f"(cross={score_cross:.2f}, line={score_line:.2f})"
        )

    print(f"\n{net.status()}")


if __name__ == "__main__":
    run()

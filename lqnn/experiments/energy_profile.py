# Quantum-inspired principle: network branch quality can be evaluated via
# free-energy-like metrics tied to structural efficiency and utilization.

from __future__ import annotations

import numpy as np

from lqnn.core.network import LiquidNetwork
from lqnn.core.quantum_state import TopologySuperposition


def run(branches: int = 4, steps: int = 30) -> list[float]:
    """Track free-energy values of topology branches over time."""
    net = LiquidNetwork(initial_neurons=14, max_neurons=220)
    sup = TopologySuperposition(net, n_branches=branches)

    last_energies: list[float] = []
    for _ in range(steps):
        signal = np.random.rand(23)
        sup.evolve(signal)
        last_energies = [sup.free_energy(branch) for branch in sup.branches]

    best, idx = sup.collapse()
    printable = [f"{v:.3f}" if np.isfinite(v) else "inf" for v in last_energies]
    print("=== Energy Profile ===")
    print(f"Energias finais por ramo: {printable}")
    print(f"Ramo escolhido: {idx}")
    print(best.status())
    return last_energies


if __name__ == "__main__":
    run()

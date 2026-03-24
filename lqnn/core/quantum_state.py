# Quantum principle: superposition of candidate topologies before collapsing
# into the most energetically efficient branch.

from __future__ import annotations

from copy import deepcopy
from typing import Tuple

import numpy as np

from lqnn.core.network import LiquidNetwork


class TopologySuperposition:
    """Explore multiple topology branches and collapse to best free energy."""

    def __init__(self, base_network: LiquidNetwork, n_branches: int = 4) -> None:
        if n_branches < 2:
            raise ValueError("n_branches deve ser >= 2")
        self.n_branches = n_branches
        self.branches: list[LiquidNetwork] = [deepcopy(base_network) for _ in range(n_branches)]
        self.branch_weights = np.ones(n_branches, dtype=float) / float(n_branches)

    def free_energy(self, net: LiquidNetwork) -> float:
        """Compute a free-energy-like score from efficiency and utilization."""
        if not net.neurons or not net.synapses:
            return float("inf")

        strong = sum(1 for syn in net.synapses if syn.is_strong)
        total_syn = len(net.synapses)
        active = sum(1 for neuron in net.neurons.values() if neuron.activation_probability > 0.5)
        total_neurons = len(net.neurons)
        if total_syn == 0 or total_neurons == 0:
            return float("inf")

        efficiency = strong / total_syn
        utilization = active / total_neurons
        return float(1.0 - 0.6 * efficiency - 0.4 * utilization)

    def evolve(self, signal: np.ndarray) -> None:
        """Evolve all branches with the same signal plus small branch noise."""
        for branch in self.branches:
            noisy = signal + np.random.normal(0.0, 0.05, len(signal))
            branch.forward(np.clip(noisy, 0.0, 1.0))

    def collapse(self, verbose: bool = True) -> Tuple[LiquidNetwork, int]:
        """Collapse superposition by choosing branch with minimal free energy."""
        energies = [self.free_energy(branch) for branch in self.branches]
        best_idx = int(np.argmin(energies))
        if verbose:
            printable = [f"{energy:.3f}" if np.isfinite(energy) else "inf" for energy in energies]
            print(f"[COLAPSO QUANTICO] Energias: {printable}")
            print(f"  Ramo {best_idx} selecionado (energia={energies[best_idx]:.3f})")
        return self.branches[best_idx], best_idx

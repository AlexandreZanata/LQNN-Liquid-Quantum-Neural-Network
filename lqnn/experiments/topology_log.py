# Scientific principle: structural memory must be measurable through explicit
# topology evolution logs over time.

from __future__ import annotations

import numpy as np

from lqnn.core.network import LiquidNetwork


def run(steps: int = 80) -> dict[str, list[float]]:
    """Run and print topology evolution statistics over multiple forwards."""
    net = LiquidNetwork(initial_neurons=16, max_neurons=300)
    for _ in range(steps):
        net.forward(np.random.rand(23))
        if net.tick_count % 25 == 0:
            net.consolidate(sleep_cycles=1)

    history = net.history
    print("=== Topology Log ===")
    print(net.status())
    print(f"  Neuronios (inicio/fim): {history['neuron_count'][0]:.0f}/{history['neuron_count'][-1]:.0f}")
    print(f"  Sinapses  (inicio/fim): {history['synapse_count'][0]:.0f}/{history['synapse_count'][-1]:.0f}")
    print(f"  Peso medio final: {history['mean_weight'][-1]:.3f}")
    print(f"  Energia media final: {history['energy_total'][-1]:.3f}")
    return history


if __name__ == "__main__":
    run()

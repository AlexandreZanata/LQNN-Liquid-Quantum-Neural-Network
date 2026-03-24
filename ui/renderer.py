# Scientific principle: every pixel must map to true internal state so emergent
# behavior is observed rather than animated fiction.

from __future__ import annotations

from typing import Any

import numpy as np

from lqnn.core.network import LiquidNetwork


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def build_network_payload(
    net: LiquidNetwork,
    *,
    last_output: list[float] | None = None,
    quantum_mode: bool = False,
    branch_energies: list[float] | None = None,
    selected_branch: int | None = None,
) -> dict[str, Any]:
    """Serialize live network state for UI rendering over WebSocket."""
    neurons = []
    active_count = 0
    energies: list[float] = []

    for neuron in net.neurons.values():
        activation = float(neuron.activation_probability)
        energy = float(neuron.energy)
        if activation > 0.6:
            active_count += 1
        energies.append(energy)
        neurons.append(
            {
                "id": neuron.id,
                "x": float(neuron.x),
                "y": float(neuron.y),
                "activation_probability": activation,
                "energy": energy,
                "alive": bool(neuron.alive),
            }
        )

    synapses = [
        {
            "pre_id": syn.pre_id,
            "post_id": syn.post_id,
            "weight": float(syn.weight),
            "strong": bool(syn.is_strong),
        }
        for syn in net.synapses
        if syn.pre_id in net.neurons and syn.post_id in net.neurons
    ]

    stats = net.stats()
    metrics = {
        "tick": stats.tick,
        "neuron_count": stats.neuron_count,
        "synapse_count": stats.synapse_count,
        "strong_synapses": stats.strong_synapses,
        "mean_weight": _safe_mean([s["weight"] for s in synapses]),
        "mean_energy": _safe_mean(energies),
        "min_energy": min(energies) if energies else 0.0,
        "max_energy": max(energies) if energies else 0.0,
        "active_ratio": float(active_count / len(neurons)) if neurons else 0.0,
    }

    return {
        "type": "state",
        "neurons": neurons,
        "synapses": synapses,
        "metrics": metrics,
        "last_output": last_output or [],
        "quantum": {
            "enabled": quantum_mode,
            "branch_energies": branch_energies or [],
            "selected_branch": selected_branch,
        },
    }

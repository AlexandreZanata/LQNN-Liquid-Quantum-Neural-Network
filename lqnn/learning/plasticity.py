# Biological principle: plasticity varies by age and context, balancing
# fast adaptation with long-term stability.

from __future__ import annotations

from lqnn.core.neuron import QuantumNeuron


def modulate_signal(signal_strength: float, neuron: QuantumNeuron, region_factor: float = 1.0) -> float:
    """Scale incoming signal by neuron plasticity and region factor."""
    scaled = signal_strength * neuron.plasticity * region_factor
    return max(0.0, min(1.0, scaled))

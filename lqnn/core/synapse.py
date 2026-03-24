# Biological principle: Hebbian plasticity strengthens pathways when pre and
# post neurons co-activate, and weakens pathways with desynchronized activity.

from __future__ import annotations

import random


class HebbianSynapse:
    """Hebbian synapse with bounded strength and survival threshold."""

    MIN_WEIGHT = 0.05
    MAX_WEIGHT = 1.0

    def __init__(self, pre_neuron_id: int, post_neuron_id: int, initial_weight: float = 0.3):
        self.pre_id = pre_neuron_id
        self.post_id = post_neuron_id
        self.weight = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, initial_weight))
        self.age = 0
        self.alive = True
        self.pre_history = []
        self.post_history = []

    @property
    def is_strong(self) -> bool:
        return self.weight > 0.6

    @property
    def signal_strength(self) -> float:
        """Transmitted signal includes small quantum-like transmission noise."""
        quantum_noise = random.gauss(0.0, 0.05)
        return max(0.0, min(1.0, self.weight + quantum_noise))

    def update(self, pre_fired: bool, post_fired: bool, learning_rate: float = 0.1) -> None:
        """Update synaptic weight using a bounded Hebbian rule."""
        self.age += 1
        self.pre_history.append(float(pre_fired))
        self.post_history.append(float(post_fired))

        if len(self.pre_history) > 10:
            self.pre_history.pop(0)
            self.post_history.pop(0)

        if pre_fired and post_fired:
            delta = learning_rate * (1.0 - self.weight)
        elif pre_fired and not post_fired:
            delta = -learning_rate * 0.3 * self.weight
        elif not pre_fired and post_fired:
            delta = 0.0
        else:
            delta = -learning_rate * 0.1 * self.weight

        self.weight = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, self.weight + delta))
        if self.weight <= self.MIN_WEIGHT:
            self.alive = False

    def __repr__(self) -> str:
        status = "FORTE" if self.is_strong else "fraca"
        return (
            f"Synapse({self.pre_id}->{self.post_id}) | "
            f"peso={self.weight:.3f} [{status}] | "
            f"age={self.age}"
        )

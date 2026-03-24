# Biological principle: neurons that fire together wire together through
# positive correlation-based reinforcement.

from __future__ import annotations

from lqnn.core.synapse import HebbianSynapse


def apply_hebbian_update(
    synapse: HebbianSynapse,
    pre_fired: bool,
    post_fired: bool,
    learning_rate: float = 0.1,
) -> float:
    """Update one synapse using Hebbian dynamics and return new weight."""
    synapse.update(pre_fired=pre_fired, post_fired=post_fired, learning_rate=learning_rate)
    return synapse.weight

# Biological principle: neurons exhibit age-dependent plasticity and decay
# when not used. Quantum inspiration: state represented by amplitudes that
# collapse into active/inactive behavior upon receiving a signal.

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(slots=True)
class QuantumNeuron:
    """Neuron with a simple quantum-inspired superposition state."""

    neuron_id: int
    x: float
    y: float
    age: int = 0
    energy: float = 1.0
    alive: bool = True
    threshold: float = 0.3
    alpha: float = field(default=math.sqrt(0.5))
    beta: float = field(default=math.sqrt(0.5))

    @property
    def id(self) -> int:
        """Compatibility alias for external code that expects `id`."""
        return self.neuron_id

    @property
    def activation_probability(self) -> float:
        """Probability of firing: |beta|^2."""
        return float(self.beta**2)

    @property
    def plasticity(self) -> float:
        """Age-dependent plasticity with a lower bound for stability."""
        return max(0.1, 1.0 / (1 + self.age * 0.002))

    def receive_signal(self, signal_strength: float) -> bool:
        """Receive signal and collapse state into a firing decision."""
        rotation_angle = signal_strength * self.plasticity * math.pi / 4
        new_beta = math.sin(math.asin(self.beta) + rotation_angle)
        self.beta = min(1.0, max(0.0, new_beta))
        self.alpha = math.sqrt(max(0.0, 1.0 - self.beta**2))

        fired = self.activation_probability > self.threshold
        if fired:
            self.energy = min(1.0, self.energy + 0.1)
        else:
            self.energy -= 0.02

        return fired

    def tick(self) -> None:
        """Advance one time step with natural decay and death."""
        self.age += 1
        self.energy -= 0.005
        self.beta = self.beta * 0.99 + math.sqrt(0.5) * 0.01
        self.alpha = math.sqrt(max(0.0, 1.0 - self.beta**2))
        if self.energy <= 0.0:
            self.alive = False

    def __repr__(self) -> str:
        return (
            f"Neuron({self.id}) | "
            f"P(ativo)={self.activation_probability:.2f} | "
            f"energia={self.energy:.2f} | "
            f"plasticidade={self.plasticity:.2f} | "
            f"age={self.age}"
        )

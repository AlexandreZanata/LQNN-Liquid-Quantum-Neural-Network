# Biological principle: user stimulation perturbs a living topology that adapts
# by activity, consolidation, and structural mutation in real time.

from __future__ import annotations

import random
from typing import Any

import numpy as np

from lqnn.core.quantum_state import TopologySuperposition
from lqnn.learning.one_shot import OneShotLearner
from ui.renderer import build_network_payload


class UIController:
    """Owns network lifecycle and maps UI commands to model actions."""

    def __init__(self, max_neurons: int = 300, input_dim: int = 23) -> None:
        self.max_neurons = max_neurons
        self.input_dim = input_dim
        self.learner = OneShotLearner()
        self.learner.net.max_neurons = max_neurons

        self.quantum_mode = False
        self.n_branches = 4
        self.superposition: TopologySuperposition | None = None
        self.branch_energies: list[float] = []
        self.selected_branch: int | None = None

        self.auto_stimulate = True
        self.last_output: list[float] = []

    @property
    def network(self):
        return self.learner.net

    def reset_network(self) -> dict[str, Any]:
        self.learner = OneShotLearner()
        self.learner.net.max_neurons = self.max_neurons
        self.superposition = None
        self.branch_energies = []
        self.selected_branch = None
        self.last_output = []
        return self.snapshot()

    def toggle_quantum_mode(self, enabled: bool | None = None) -> dict[str, Any]:
        if enabled is None:
            self.quantum_mode = not self.quantum_mode
        else:
            self.quantum_mode = bool(enabled)

        self.superposition = None
        self.branch_energies = []
        self.selected_branch = None
        return self.snapshot()

    def set_auto_stimulate(self, enabled: bool) -> dict[str, Any]:
        self.auto_stimulate = bool(enabled)
        return self.snapshot()

    def stimulate_text(self, text: str) -> dict[str, Any]:
        signal = self.learner.encode(text)
        self._forward(signal)
        return self.snapshot()

    def stimulate_random(self) -> dict[str, Any]:
        signal = np.random.rand(self.input_dim)
        self._forward(signal)
        return self.snapshot()

    def stimulate_vector(self, values: list[float]) -> dict[str, Any]:
        arr = np.array(values, dtype=float)
        if len(arr) < self.input_dim:
            arr = np.pad(arr, (0, self.input_dim - len(arr)))
        elif len(arr) > self.input_dim:
            arr = arr[: self.input_dim]
        self._forward(np.clip(arr, 0.0, 1.0))
        return self.snapshot()

    def sleep(self, cycles: int = 1) -> dict[str, Any]:
        self.network.consolidate(sleep_cycles=max(1, int(cycles)))
        self._apply_spatial_dynamics(intensity=0.2)
        return self.snapshot()

    def idle_step(self) -> dict[str, Any]:
        if self.auto_stimulate:
            signal = np.random.rand(self.input_dim) * 0.25
            self._forward(signal)
        else:
            for neuron in self.network.neurons.values():
                neuron.tick()
            self.network.tick_count += 1
            self.network._prune_dead()
            self.network._log_metrics()
            self._apply_spatial_dynamics(intensity=0.08)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return build_network_payload(
            self.network,
            last_output=self.last_output,
            quantum_mode=self.quantum_mode,
            branch_energies=self.branch_energies,
            selected_branch=self.selected_branch,
        )

    def _forward(self, signal: np.ndarray) -> None:
        if self.quantum_mode:
            if self.superposition is None:
                self.superposition = TopologySuperposition(self.network, n_branches=self.n_branches)
            self.superposition.evolve(signal)
            self.branch_energies = [
                self.superposition.free_energy(branch) for branch in self.superposition.branches
            ]
            best, idx = self.superposition.collapse(verbose=False)
            self.learner.net = best
            self.selected_branch = idx
            self.superposition = TopologySuperposition(best, n_branches=self.n_branches)
            output = best.forward(signal)
        else:
            output = self.network.forward(signal)

        self.last_output = output.tolist()
        self._apply_spatial_dynamics(intensity=0.35)

    def _apply_spatial_dynamics(self, intensity: float) -> None:
        for neuron in self.network.neurons.values():
            drift_scale = intensity * max(0.1, neuron.plasticity)
            dx = (neuron.activation_probability - 0.5) * drift_scale + random.uniform(-0.2, 0.2) * drift_scale
            dy = (neuron.energy - 0.5) * drift_scale + random.uniform(-0.2, 0.2) * drift_scale
            neuron.x = min(self.network.space_size - 2.0, max(2.0, neuron.x + dx))
            neuron.y = min(self.network.space_size - 2.0, max(2.0, neuron.y + dy))

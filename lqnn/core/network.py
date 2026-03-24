# Biological principle: a living neural graph changes topology continuously:
# active structures are reinforced, while inactive elements are pruned.

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np

from lqnn.core.neuron import QuantumNeuron
from lqnn.core.synapse import HebbianSynapse


@dataclass(slots=True)
class NetworkStats:
    neuron_count: int
    synapse_count: int
    strong_synapses: int
    tick: int


class LiquidNetwork:
    """Dynamic network that can create and prune neurons and synapses."""

    def __init__(self, initial_neurons: int = 12, max_neurons: int = 500, space_size: float = 100.0):
        self.max_neurons = max_neurons
        self.space_size = space_size
        self.neurons: dict[int, QuantumNeuron] = {}
        self.synapses: list[HebbianSynapse] = []
        self._next_id = 0
        self.tick_count = 0
        self.history: dict[str, list[float]] = {
            "neuron_count": [],
            "synapse_count": [],
            "mean_weight": [],
            "energy_total": [],
        }
        self._initialize(initial_neurons)

    def _initialize(self, n: int) -> None:
        for _ in range(n):
            self._spawn_neuron(
                x=random.uniform(10.0, self.space_size - 10.0),
                y=random.uniform(10.0, self.space_size - 10.0),
            )

        ids = list(self.neurons.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = self.neurons[ids[i]]
                b = self.neurons[ids[j]]
                distance = math.hypot(a.x - b.x, a.y - b.y)
                if distance < 30.0 and random.random() > 0.4:
                    self.synapses.append(
                        HebbianSynapse(ids[i], ids[j], initial_weight=random.uniform(0.2, 0.6))
                    )

    def _spawn_neuron(self, x: float, y: float, parent_id: int | None = None) -> int:
        nid = self._next_id
        self._next_id += 1
        self.neurons[nid] = QuantumNeuron(neuron_id=nid, x=x, y=y)

        if parent_id is not None and parent_id in self.neurons:
            self.synapses.append(HebbianSynapse(parent_id, nid, initial_weight=0.4))
        return nid

    def _prune_dead(self) -> None:
        dead_neurons = {nid for nid, neuron in self.neurons.items() if not neuron.alive}
        self.neurons = {nid: neuron for nid, neuron in self.neurons.items() if neuron.alive}
        self.synapses = [
            syn
            for syn in self.synapses
            if syn.alive and syn.pre_id not in dead_neurons and syn.post_id not in dead_neurons
        ]

    def forward(self, input_signal: np.ndarray) -> np.ndarray:
        """Propagate signal and adapt topology as side effect."""
        live_ids = list(self.neurons.keys())
        if not live_ids:
            return np.array([], dtype=float)

        fired: dict[int, bool] = {}
        n_input = min(len(input_signal), len(live_ids))
        for i, nid in enumerate(live_ids[:n_input]):
            fired[nid] = self.neurons[nid].receive_signal(float(input_signal[i]))

        for syn in self.synapses:
            if not syn.alive:
                continue
            if syn.pre_id in fired and fired[syn.pre_id]:
                if syn.post_id in self.neurons:
                    post_fired = self.neurons[syn.post_id].receive_signal(syn.signal_strength)
                    fired[syn.post_id] = post_fired
                    syn.update(pre_fired=True, post_fired=post_fired)
                else:
                    syn.update(pre_fired=True, post_fired=False)
            elif syn.pre_id in fired:
                syn.update(pre_fired=False, post_fired=fired.get(syn.post_id, False))

        active_ids = [nid for nid, is_on in fired.items() if is_on]
        self._maybe_create_synapses(active_ids)

        if active_ids and len(self.neurons) < self.max_neurons and random.random() < 0.15:
            parent = random.choice(active_ids)
            p = self.neurons[parent]
            angle = random.uniform(0.0, 2.0 * math.pi)
            distance = random.uniform(8.0, 20.0)
            nx = min(self.space_size - 5.0, max(5.0, p.x + math.cos(angle) * distance))
            ny = min(self.space_size - 5.0, max(5.0, p.y + math.sin(angle) * distance))
            self._spawn_neuron(nx, ny, parent_id=parent)

        for neuron in self.neurons.values():
            neuron.tick()

        self._prune_dead()
        self.tick_count += 1
        self._log_metrics()
        return np.array([float(fired.get(nid, False)) for nid in self.neurons.keys()], dtype=float)

    def _maybe_create_synapses(self, active_ids: list[int]) -> None:
        existing = {(syn.pre_id, syn.post_id) for syn in self.synapses}
        for i in range(len(active_ids)):
            for j in range(i + 1, len(active_ids)):
                a, b = active_ids[i], active_ids[j]
                if (a, b) in existing or (b, a) in existing:
                    continue
                na = self.neurons.get(a)
                nb = self.neurons.get(b)
                if na is None or nb is None:
                    continue
                if math.hypot(na.x - nb.x, na.y - nb.y) < 35.0:
                    self.synapses.append(HebbianSynapse(a, b, initial_weight=0.25))
                    existing.add((a, b))

    def consolidate(self, sleep_cycles: int = 3) -> None:
        """Sleep-like consolidation phase: strengthen useful, prune weak."""
        print(f"\n[CONSOLIDACAO] Iniciando {sleep_cycles} ciclos de sono...")
        for _ in range(sleep_cycles):
            for syn in self.synapses:
                if syn.weight > 0.6:
                    syn.weight = min(1.0, syn.weight * 1.05)
            for syn in self.synapses:
                if syn.weight < 0.3:
                    syn.weight *= 0.85
            self.synapses = [syn for syn in self.synapses if syn.weight > HebbianSynapse.MIN_WEIGHT]

            connected = {syn.pre_id for syn in self.synapses} | {syn.post_id for syn in self.synapses}
            for nid, neuron in self.neurons.items():
                if nid not in connected and neuron.age > 50:
                    neuron.energy -= 0.3
            self._prune_dead()

        print(
            f"[CONSOLIDACAO] Rede apos sono: {len(self.neurons)} neuronios, "
            f"{len(self.synapses)} sinapses"
        )

    def _log_metrics(self) -> None:
        live = list(self.neurons.values())
        self.history["neuron_count"].append(float(len(live)))
        self.history["synapse_count"].append(float(len(self.synapses)))
        self.history["mean_weight"].append(
            float(np.mean([syn.weight for syn in self.synapses])) if self.synapses else 0.0
        )
        self.history["energy_total"].append(
            float(np.mean([neuron.energy for neuron in live])) if live else 0.0
        )

    def stats(self) -> NetworkStats:
        strong = sum(1 for syn in self.synapses if syn.is_strong)
        return NetworkStats(
            neuron_count=len(self.neurons),
            synapse_count=len(self.synapses),
            strong_synapses=strong,
            tick=self.tick_count,
        )

    def status(self) -> str:
        s = self.stats()
        return (
            f"Rede | {s.neuron_count} neuronios | {s.synapse_count} sinapses "
            f"({s.strong_synapses} fortes) | tick={s.tick}"
        )

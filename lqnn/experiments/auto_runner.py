# Scientific principle: continuous stimulation and consolidation cycles expose
# rates of structural birth/death and long-term stability of memory topology.

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from lqnn.learning.one_shot import OneShotLearner


@dataclass(slots=True)
class StepLog:
    tick: int
    mode: str
    neuron_count: int
    synapse_count: int
    births: int
    deaths: int
    synapse_delta: int
    active_ratio: float
    mean_energy: float
    mean_weight: float
    stability_score: float


def _compute_stability(prev_neurons: int, prev_synapses: int, cur_neurons: int, cur_synapses: int) -> float:
    if prev_neurons == 0 and prev_synapses == 0:
        return 1.0
    delta = abs(cur_neurons - prev_neurons) + abs(cur_synapses - prev_synapses)
    denom = max(prev_neurons + prev_synapses, 1)
    return float(max(0.0, 1.0 - (delta / denom)))


def run(steps: int = 200, sleep_interval: int = 15) -> Path:
    """Run automated stimulation/sleep cycles and persist JSON logs."""
    learner = OneShotLearner()
    net = learner.net
    input_dim = 23

    prev_ids = set(net.neurons.keys())
    prev_synapses = len(net.synapses)
    prev_neurons = len(net.neurons)

    rows: list[StepLog] = []
    text_patterns = [
        "excelente, gostei bastante",
        "horrivel, nao presta",
        "qualidade muito boa",
        "decepcao total",
    ]

    for step in range(steps):
        if step % sleep_interval == 0 and step > 0:
            mode = "sleep"
            net.consolidate(sleep_cycles=1)
        elif step % 2 == 0:
            mode = "random"
            net.forward(np.random.rand(input_dim))
        else:
            mode = "structured"
            signal = learner.encode(text_patterns[step % len(text_patterns)])
            net.forward(signal)

        cur_ids = set(net.neurons.keys())
        births = len(cur_ids - prev_ids)
        deaths = len(prev_ids - cur_ids)

        cur_synapses = len(net.synapses)
        cur_neurons = len(net.neurons)
        synapse_delta = cur_synapses - prev_synapses

        active_ratio = 0.0
        mean_energy = 0.0
        if net.neurons:
            activations = [n.activation_probability for n in net.neurons.values()]
            energies = [n.energy for n in net.neurons.values()]
            active_ratio = float(sum(a > 0.6 for a in activations) / len(activations))
            mean_energy = float(np.mean(energies))

        mean_weight = float(np.mean([s.weight for s in net.synapses])) if net.synapses else 0.0
        stability = _compute_stability(prev_neurons, prev_synapses, cur_neurons, cur_synapses)

        rows.append(
            StepLog(
                tick=net.tick_count,
                mode=mode,
                neuron_count=cur_neurons,
                synapse_count=cur_synapses,
                births=births,
                deaths=deaths,
                synapse_delta=synapse_delta,
                active_ratio=active_ratio,
                mean_energy=mean_energy,
                mean_weight=mean_weight,
                stability_score=stability,
            )
        )

        prev_ids = cur_ids
        prev_synapses = cur_synapses
        prev_neurons = cur_neurons

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = logs_dir / f"run_{timestamp}.json"

    payload = {
        "metadata": {
            "steps": steps,
            "sleep_interval": sleep_interval,
            "final_status": net.status(),
        },
        "timeline": [asdict(row) for row in rows],
    }

    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Log salvo em: {output}")
    print(net.status())
    return output


if __name__ == "__main__":
    run()

# Biological principle: topology and activity should be directly observable,
# enabling immediate feedback about structural adaptation.

from __future__ import annotations

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist

from lqnn.learning.one_shot import OneShotLearner


def run_live_visualization(frames: int = 200, interval_ms: int = 150) -> None:
    """Render the living topology and key network metrics in real time."""
    learner = OneShotLearner()
    net = learner.net

    fig, (ax_net, ax_metrics) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0a0a0a")

    examples = [
        ("positivo", "excelente, adorei muito"),
        ("negativo", "pessimo, horrivel"),
        ("positivo", "muito bom produto"),
        ("negativo", "ruim e caro demais"),
        ("positivo", "recomendo fortemente"),
    ]
    example_idx = 0
    neuron_history: list[int] = []
    synapse_history: list[int] = []

    def update(frame: int) -> list[Artist]:
        nonlocal example_idx
        ax_net.clear()
        ax_metrics.clear()

        ax_net.set_facecolor("#0a0a0a")
        ax_net.set_xlim(0, net.space_size)
        ax_net.set_ylim(0, net.space_size)
        ax_net.set_title("LQNN - Topologia Viva", color="white", fontsize=12)
        ax_net.tick_params(colors="#444")

        if frame % 8 == 0 and example_idx < len(examples):
            label, ex = examples[example_idx]
            learner.learn(label, ex)
            example_idx += 1

        if frame % 40 == 0 and frame > 0:
            net.consolidate(sleep_cycles=1)

        signal = np.random.rand(23) * 0.3
        net.forward(signal)

        for syn in net.synapses:
            if syn.pre_id not in net.neurons or syn.post_id not in net.neurons:
                continue
            pre = net.neurons[syn.pre_id]
            post = net.neurons[syn.post_id]
            alpha = 0.1 + syn.weight * 0.6
            line_width = syn.weight * 2.5
            color = "#4B6CB7" if syn.is_strong else "#445066"
            ax_net.plot(
                [pre.x, post.x],
                [pre.y, post.y],
                color=color,
                alpha=alpha,
                linewidth=line_width,
                zorder=1,
            )

        for neuron in net.neurons.values():
            size = 30 + neuron.activation_probability * 120
            alpha = 0.4 + neuron.energy * 0.6
            color = (
                "#1EA36B"
                if neuron.activation_probability > 0.6
                else "#2E86C1"
                if neuron.energy > 0.5
                else "#8C8C84"
            )
            ax_net.scatter(neuron.x, neuron.y, s=size, c=color, alpha=alpha, zorder=2)

        ax_net.set_xlabel(
            f"Neuronios: {len(net.neurons)} | Sinapses: {len(net.synapses)} | Tick: {net.tick_count}",
            color="#888",
            fontsize=9,
        )

        neuron_history.append(len(net.neurons))
        synapse_history.append(len(net.synapses))

        ax_metrics.set_facecolor("#0a0a0a")
        ax_metrics.set_title("Evolucao da topologia", color="white", fontsize=12)
        if neuron_history:
            ax_metrics.plot(neuron_history, color="#2E86C1", label="neuronios", linewidth=1.5)
        if synapse_history:
            ax_metrics.plot(synapse_history, color="#4B6CB7", label="sinapses", linewidth=1.5)
        ax_metrics.legend(facecolor="#111", labelcolor="white", fontsize=9)
        ax_metrics.tick_params(colors="#555")
        ax_metrics.set_xlabel("ticks", color="#666", fontsize=9)
        return []

    _ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval_ms, repeat=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_live_visualization()

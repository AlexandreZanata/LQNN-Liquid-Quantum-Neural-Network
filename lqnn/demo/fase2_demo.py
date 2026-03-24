# Biological principle: co-activation strengthens synapses, while persistent
# mismatch weakens pathways and may eliminate unused connections.

from __future__ import annotations

from lqnn.core.neuron import QuantumNeuron
from lqnn.core.synapse import HebbianSynapse


def run() -> None:
    print("=== LQNN - Fase 2: Sinapses Hebbianas ===\n")

    pre = QuantumNeuron(0, 0.0, 0.0)
    post = QuantumNeuron(1, 1.0, 0.0)
    syn = HebbianSynapse(pre.id, post.id, initial_weight=0.3)

    print("Treinando: pre e pos disparam juntos 10x")
    for i in range(10):
        pre_fired = pre.receive_signal(0.9)
        post_fired = post.receive_signal(syn.signal_strength)
        syn.update(pre_fired=pre_fired, post_fired=post_fired)
        if i % 3 == 0:
            print(f"  iteracao {i + 1}: {syn}")

    print(f"\nSinapse apos co-ativacao: peso={syn.weight:.3f}")

    print("\nQuebrando associacao: pre dispara, pos nao recebe sinal")
    for _ in range(15):
        pre_fired = pre.receive_signal(0.9)
        post_fired = False
        syn.update(pre_fired=pre_fired, post_fired=post_fired)

    print(f"Sinapse apos dissociacao: peso={syn.weight:.3f} | viva={syn.alive}")


if __name__ == "__main__":
    run()

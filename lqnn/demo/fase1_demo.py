# Biological principle: repeated stimulation reinforces activation tendency,
# while inactivity causes decay and eventual death of underused neurons.

from __future__ import annotations

from lqnn.core.neuron import QuantumNeuron


def run() -> None:
    n = QuantumNeuron(neuron_id=0, x=0.0, y=0.0)
    print("=== LQNN - Fase 1: Neuronio Vivo ===\n")
    print(f"Nascimento: {n}")

    print("\n--- Estimulando com sinal forte (aprendizado) ---")
    for i in range(5):
        fired = n.receive_signal(signal_strength=0.8)
        print(f"  tick {i + 1}: disparou={fired} | {n}")

    print("\n--- Sem estimulo (esquecimento) ---")
    for _ in range(20):
        n.tick()

    print(f"  apos 20 ticks sem uso: {n}")

    print("\n--- Estimulacao fraca (nao suficiente) ---")
    for _ in range(10):
        _ = n.receive_signal(signal_strength=0.1)
        n.tick()

    print(f"  resultado: {n}")
    print(f"\n  Neuronio ainda vivo? {n.alive}")


if __name__ == "__main__":
    run()

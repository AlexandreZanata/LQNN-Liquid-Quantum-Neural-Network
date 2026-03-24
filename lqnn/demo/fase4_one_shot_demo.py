# Learning principle: one-shot categorization emerges from structural memory
# snapshots rather than multi-epoch gradient optimization.

from __future__ import annotations

from lqnn.learning.one_shot import OneShotLearner


def run() -> None:
    print("=" * 55)
    print("LQNN - One-Shot Learning sem treinamento")
    print("=" * 55)

    learner = OneShotLearner()

    learner.learn("positivo", "esse produto e incrivel, adorei muito")
    learner.learn("negativo", "pessimo produto, nao funciona, horrivel")

    learner.net.consolidate(sleep_cycles=2)

    testes = [
        "que experiencia maravilhosa, recomendo",
        "terrivel, nunca mais compro isso",
        "gostei bastante, muito bom mesmo",
        "produto ruim, decepcionante demais",
        "excelente qualidade, surpreendente",
        "nao presta, jogou no lixo",
    ]

    print("\n--- Classificando frases novas ---")
    corretos = 0
    for frase in testes:
        label, score = learner.classify(frase)
        esperado = (
            "positivo"
            if any(token in frase for token in ["maravilhosa", "bom", "gostei", "excelente"])
            else "negativo"
        )
        ok = "OK" if label == esperado else "X "
        if label == esperado:
            corretos += 1
        print(f"  [{ok}] '{frase[:40]}...' -> {label} (score={score:.2f})")

    acc = corretos / len(testes)
    print(f"\nAcuracia: {corretos}/{len(testes)} = {acc * 100:.0f}%")
    print("Treinamento usado: 1 exemplo por classe")
    print("Epochs: 0  |  Backprop: nenhum  |  GPU: nao necessaria")
    print(f"\n{learner.net.status()}")


if __name__ == "__main__":
    run()

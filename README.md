# LQNN - Liquid Quantum Neural Network

Arquitetura onde a estrutura da rede representa a memoria.

## Estado atual

- Arquitetura profissional inicial criada com separacao por dominios: `core`, `learning`, `demo`, `experiments`.
- Fase 1 implementada: `QuantumNeuron` com plasticidade por idade, decaimento e morte por inatividade.
- Fase 2 implementada: `HebbianSynapse` com reforco por co-ativacao e enfraquecimento por dissociacao.
- Fase 3 implementada: `LiquidNetwork` com crescimento e poda dinamicos por atividade/energia.
- Baseline open source baixado para testes em `models/open_source/tiny-gpt2`.

## Estrutura

```text
lqnn/
  core/
  learning/
  demo/
  experiments/
```

## Ambiente

```bash
python3 -m venv .venv
.venv/bin/pip install numpy matplotlib tqdm
```

## Como executar

```bash
.venv/bin/python demo/fase1_demo.py
.venv/bin/python demo/fase2_demo.py
.venv/bin/python -c "from core.network import LiquidNetwork; n=LiquidNetwork(); print(n.status())"
```

## Download de modelo open source para testes

```bash
.venv/bin/python experiments/download_open_model.py
```

## Proximos passos

1. Implementar Fase 4 (`learning/one_shot.py`) com snapshots topologicos por classe.
2. Criar demo de classificacao one-shot (`demo/fase4_one_shot_demo.py`).
3. Seguir sequencia do arquivo de diretrizes: `begining.md`.

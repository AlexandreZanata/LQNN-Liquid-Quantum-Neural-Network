# LQNN - Liquid Quantum Neural Network

Arquitetura onde a estrutura da rede representa a memoria.

## Estado atual

- Arquitetura profissional inicial criada com separacao por dominios: `core`, `learning`, `demo`, `experiments`.
- Fase 1 implementada: `QuantumNeuron` com plasticidade por idade, decaimento e morte por inatividade.
- Fase 2 implementada: `HebbianSynapse` com reforco por co-ativacao e enfraquecimento por dissociacao.
- Fase 3 implementada: `LiquidNetwork` com crescimento e poda dinamicos por atividade/energia.
- Fase 4 implementada: `OneShotLearner` com memoria estrutural por snapshots (`learning/one_shot.py`).
- Fase 5 implementada: visualizacao em tempo real da topologia (`demo/visualizer.py`).
- Fase 6 implementada: superposicao de topologias com colapso por energia livre (`core/quantum_state.py`).
- Fase 7 base implementada: UI em tempo real com FastAPI + WebSocket + Canvas (`ui/app.py`).
- Fase 8 base implementada: painel de interacao com comandos de estimulo/sono/reset/quantico.
- Fase 9 implementada: executor automatico com log JSON (`experiments/auto_runner.py`).
- Experimentos praticos ativos: `experiments/topology_log.py`, `experiments/energy_profile.py`, `experiments/vs_mlp.py`.

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
.venv/bin/python demo/fase4_one_shot_demo.py
.venv/bin/python demo/visualizer.py
.venv/bin/python -c "from core.network import LiquidNetwork; from core.quantum_state import TopologySuperposition; import numpy as np; net=LiquidNetwork(); sup=TopologySuperposition(net, n_branches=4); sup.evolve(np.random.rand(23)); best, idx=sup.collapse(); print(best.status(), idx)"
.venv/bin/python -m uvicorn ui.app:app --host 0.0.0.0 --port 8000
```

## Testes praticos

```bash
.venv/bin/python experiments/topology_log.py
.venv/bin/python experiments/energy_profile.py
.venv/bin/python experiments/vs_mlp.py
.venv/bin/python experiments/auto_runner.py
```

## Download de modelo open source para testes

```bash
.venv/bin/python experiments/download_open_model.py
```

## Proximos passos

1. Refinar criterio de similaridade topologica da Fase 4 para reduzir empates de score.
2. Integrar logs de energia/topologia em CSV para analise longitudinal.
3. Seguir sequencia de diretrizes e experimentos em `begining.md`.

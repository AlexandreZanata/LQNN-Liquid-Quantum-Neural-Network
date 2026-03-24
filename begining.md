# LQNN — Liquid Quantum Neural Network
## Prompt completo para o Cursor — estruturado em fases de 2 horas

---

## VISÃO GERAL DO PROJETO

Você vai me ajudar a construir o **LQNN** — uma arquitetura de rede neural onde a **estrutura é a memória**. Diferente de todo modelo existente (GPT, BERT, LLaMA), o LQNN não tem topologia fixa. Ele aprende com 1 único exemplo mudando quem vive, quem morre e quem nasce na rede — exatamente como o cérebro biológico.

**Princípio central:**
> Em redes neurais convencionais, os pesos mudam mas a estrutura é estática. No LQNN, os pesos são secundários — a topologia (quais neurônios existem e como estão conectados) é o conhecimento. Aprender = mudar a estrutura.

**Três inovações que nenhuma arquitetura combinou:**
1. Morte e nascimento de neurônios durante INFERÊNCIA (não treinamento)
2. Superposição topológica quântica — a rede explora N estruturas em paralelo antes de colapsar para a mais eficiente
3. Consolidação por inatividade — ciclos de "sono" que podas neurônios não usados e fortalecem os usados

**Hardware alvo:** RTX 4060 8GB · 32GB RAM · i7 13ª geração
**Sem treinamento longo. Sem datasets externos. Resultado visível a cada fase.**

---

## ESTRUTURA DO PROJETO

```
lqnn/
├── core/
│   ├── neuron.py              # Neurônio com estado quântico (superposição)
│   ├── synapse.py             # Sinapse com peso + força hebbiana
│   ├── network.py             # Grafo vivo — gerencia nascimento/morte
│   ├── quantum_state.py       # Superposição topológica
│   └── consolidation.py      # Ciclo de "sono" — poda e fortalecimento
│
├── learning/
│   ├── one_shot.py            # Aprendizado com 1 exemplo
│   ├── hebbian.py             # Regra hebbiana: "neurons that fire together wire together"
│   └── plasticity.py         # Controle de plasticidade por região da rede
│
├── demo/
│   ├── visualizer.py          # Visualização em tempo real com pygame ou matplotlib
│   ├── text_demo.py           # Demo: classificar frases com 1 exemplo
│   └── pattern_demo.py        # Demo: reconhecer padrões visuais simples
│
├── experiments/
│   ├── vs_mlp.py              # Comparar LQNN vs MLP clássico em one-shot
│   ├── topology_log.py        # Logar evolução da topologia ao longo do tempo
│   └── energy_profile.py      # Medir energia livre da rede em cada estado
│
├── requirements.txt
└── README.md
```

---

## FASE 1 — O NEURÔNIO VIVO (2 horas)
### Objetivo: ver um único neurônio nascer, pulsar e morrer

**O que construir:** `core/neuron.py` e um script de demo simples.

```python
# core/neuron.py

import numpy as np

class QuantumNeuron:
    """
    Neurônio com estado quântico de superposição.

    Estado: vetor [alpha, beta] onde
      alpha = amplitude de estar INATIVO (|0⟩)
      beta  = amplitude de estar ATIVO   (|1⟩)
      |alpha|² + |beta|² = 1 sempre

    Antes de receber input, o neurônio existe em superposição.
    Ao receber input (medição), colapsa para ativo ou inativo
    com probabilidade |beta|².

    Plasticidade: neurônios jovens têm alta plasticidade (mudam fácil).
    Neurônios velhos têm baixa plasticidade (resistem a mudanças).
    Isso é exatamente o que o cérebro faz.
    """

    def __init__(self, neuron_id: int, x: float, y: float):
        self.id = neuron_id
        self.x = x                          # posição no espaço 2D
        self.y = y
        self.age = 0                        # ticks de vida
        self.energy = 1.0                   # energia atual (0 = morte)
        self.alive = True

        # Estado quântico inicial — superposição uniforme
        self.alpha = np.sqrt(0.5)           # amplitude inativo
        self.beta  = np.sqrt(0.5)           # amplitude ativo

        # Limiar de ativação (aumenta com a idade — neurônios velhos são mais seletivos)
        self.threshold = 0.3

    @property
    def activation_probability(self) -> float:
        """Probabilidade de estar ativo = |beta|²"""
        return float(self.beta ** 2)

    @property
    def plasticity(self) -> float:
        """
        Plasticidade diminui com a idade.
        Neurônio novo (age=0): plasticidade = 1.0
        Neurônio velho (age=1000): plasticidade ≈ 0.1
        """
        return max(0.1, 1.0 / (1 + self.age * 0.002))

    def receive_signal(self, signal_strength: float) -> bool:
        """
        Recebe sinal e colapsa o estado quântico.
        Retorna True se o neurônio disparou (ativou).
        """
        # Atualiza estado quântico baseado no sinal
        rotation_angle = signal_strength * self.plasticity * np.pi / 4
        new_beta = np.sin(np.arcsin(self.beta) + rotation_angle)
        new_beta = np.clip(new_beta, 0, 1)
        self.beta = new_beta
        self.alpha = np.sqrt(max(0, 1 - self.beta**2))

        # Colapso: dispara se probabilidade > limiar
        fired = self.activation_probability > self.threshold

        if fired:
            self.energy = min(1.0, self.energy + 0.1)  # uso mantém energia
        else:
            self.energy -= 0.02  # desuso drena energia

        return fired

    def tick(self):
        """
        Avança 1 unidade de tempo.
        Neurônios que não são usados perdem energia e morrem.
        """
        self.age += 1
        self.energy -= 0.005  # decay natural

        # Decaimento quântico — sem input, volta para superposição
        self.beta = self.beta * 0.99 + np.sqrt(0.5) * 0.01
        self.alpha = np.sqrt(max(0, 1 - self.beta**2))

        if self.energy <= 0:
            self.alive = False

    def __repr__(self):
        return (f"Neuron({self.id}) | "
                f"P(ativo)={self.activation_probability:.2f} | "
                f"energia={self.energy:.2f} | "
                f"plasticidade={self.plasticity:.2f} | "
                f"age={self.age}")
```

**Script de validação da Fase 1** — rode isso e veja o neurônio viver:

```python
# demo/fase1_demo.py

from core.neuron import QuantumNeuron
import time

n = QuantumNeuron(neuron_id=0, x=0, y=0)
print("=== LQNN — Fase 1: Neurônio Vivo ===\n")
print(f"Nascimento: {n}")

print("\n--- Estimulando com sinal forte (aprendizado) ---")
for i in range(5):
    fired = n.receive_signal(signal_strength=0.8)
    print(f"  tick {i+1}: disparou={fired} | {n}")

print("\n--- Sem estimulação (esquecimento) ---")
for i in range(20):
    n.tick()

print(f"  após 20 ticks sem uso: {n}")

print("\n--- Estimulação fraca (não suficiente) ---")
for i in range(10):
    fired = n.receive_signal(signal_strength=0.1)
    n.tick()

print(f"  resultado: {n}")
print(f"\n  Neurônio ainda vivo? {n.alive}")
```

**Resultado esperado:** você verá o neurônio aumentar sua probabilidade de ativação com estímulos fortes, decair sem uso, e eventualmente morrer se abandonado. Isso é neuroplasticidade implementada.

---

## FASE 2 — A SINAPSE HEBBIANA (2 horas)
### Objetivo: dois neurônios se conectarem automaticamente quando ativados juntos

```python
# core/synapse.py

import numpy as np

class HebbianSynapse:
    """
    Sinapse que implementa a Regra de Hebb:
    "Neurons that fire together, wire together."
    "Neurons that fire apart, wire apart."

    Peso aumenta quando pré E pós disparam juntos.
    Peso diminui quando disparam em momentos diferentes.
    Sinapse morre quando peso cai abaixo do threshold.
    """

    MIN_WEIGHT = 0.05   # abaixo disso → sinapse morre
    MAX_WEIGHT = 1.0

    def __init__(self, pre_neuron_id: int, post_neuron_id: int,
                 initial_weight: float = 0.3):
        self.pre_id  = pre_neuron_id
        self.post_id = post_neuron_id
        self.weight  = initial_weight
        self.age     = 0
        self.alive   = True

        # Histórico de co-ativações (para calcular correlação)
        self.pre_history  = []
        self.post_history = []

    @property
    def is_strong(self) -> bool:
        return self.weight > 0.6

    @property
    def signal_strength(self) -> float:
        """Sinal transmitido = peso × probabilidade de transmissão quântica"""
        quantum_noise = np.random.normal(0, 0.05)
        return np.clip(self.weight + quantum_noise, 0, 1)

    def update(self, pre_fired: bool, post_fired: bool,
               learning_rate: float = 0.1):
        """
        Atualiza peso pela regra de Hebb modificada.
        Inclui termo de decaimento para evitar saturação (biológicamente realista).
        """
        self.age += 1
        self.pre_history.append(float(pre_fired))
        self.post_history.append(float(post_fired))

        # Guarda apenas os últimos 10 eventos
        if len(self.pre_history) > 10:
            self.pre_history.pop(0)
            self.post_history.pop(0)

        # Regra de Hebb: correlação entre pré e pós
        if pre_fired and post_fired:
            # Co-ativação: fortalece
            delta = learning_rate * (1 - self.weight)  # bounded growth
        elif pre_fired and not post_fired:
            # Pré disparou mas pós não: enfraquece levemente
            delta = -learning_rate * 0.3 * self.weight
        elif not pre_fired and post_fired:
            # Pós disparou sem pré: neutro (não causal)
            delta = 0.0
        else:
            # Nenhum disparou: decaimento lento (forgetting)
            delta = -learning_rate * 0.1 * self.weight

        self.weight = np.clip(self.weight + delta,
                              self.MIN_WEIGHT, self.MAX_WEIGHT)

        if self.weight <= self.MIN_WEIGHT:
            self.alive = False

    def __repr__(self):
        status = "FORTE" if self.is_strong else "fraca"
        return (f"Synapse({self.pre_id}→{self.post_id}) | "
                f"peso={self.weight:.3f} [{status}] | "
                f"age={self.age}")
```

**Script de validação da Fase 2:**

```python
# demo/fase2_demo.py

from core.neuron import QuantumNeuron
from core.synapse import HebbianSynapse

print("=== LQNN — Fase 2: Sinapses Hebbianas ===\n")

pre  = QuantumNeuron(0, 0, 0)
post = QuantumNeuron(1, 1, 0)
syn  = HebbianSynapse(pre.pre_id if hasattr(pre,'pre_id') else 0,
                      post.id, initial_weight=0.3)

print("Treinando: pré e pós disparam juntos 10x")
for i in range(10):
    pre_fired  = pre.receive_signal(0.9)
    post_fired = post.receive_signal(syn.signal_strength)
    syn.update(pre_fired, post_fired)
    if i % 3 == 0:
        print(f"  iteração {i+1}: {syn}")

print(f"\nSinapse após co-ativação: peso={syn.weight:.3f}")

print("\nQuebrando a associação: pré dispara, pós não recebe sinal")
for i in range(15):
    pre_fired  = pre.receive_signal(0.9)
    post_fired = post.receive_signal(0.0)  # sem sinal
    syn.update(pre_fired, post_fired)

print(f"Sinapse após dissociação: peso={syn.weight:.3f} | viva={syn.alive}")
```

---

## FASE 3 — A REDE VIVA (2 horas)
### Objetivo: rede que cresce e encolhe sozinha, visível em tempo real

```python
# core/network.py

import numpy as np
from typing import List, Dict, Tuple, Optional
from core.neuron import QuantumNeuron
from core.synapse import HebbianSynapse

class LiquidNetwork:
    """
    Rede neural líquida — topologia muda a cada inferência.

    Regras de vida:
    - Neurônio nasce: quando um neurônio ativo não consegue propagar
      seu sinal por sinapses existentes (necessidade não atendida)
    - Neurônio morre: quando energia cai a zero (desuso prolongado)
    - Sinapse nasce: entre dois neurônios que disparam juntos sem
      conexão direta (detecção de co-ativação)
    - Sinapse morre: quando peso cai abaixo do mínimo (desuso)

    Isso é exatamente neuroplasticidade: use it or lose it.
    """

    def __init__(self, initial_neurons: int = 12,
                 max_neurons: int = 500,
                 space_size: float = 100.0):
        self.max_neurons = max_neurons
        self.space_size  = space_size
        self.neurons: Dict[int, QuantumNeuron] = {}
        self.synapses: List[HebbianSynapse] = []
        self._next_id = 0
        self.tick_count = 0

        # Métricas para acompanhar evolução
        self.history = {
            'neuron_count': [],
            'synapse_count': [],
            'mean_weight': [],
            'energy_total': [],
        }

        self._initialize(initial_neurons)

    def _initialize(self, n: int):
        for _ in range(n):
            self._spawn_neuron(
                x=np.random.uniform(10, self.space_size - 10),
                y=np.random.uniform(10, self.space_size - 10)
            )
        # Conectar neurônios próximos inicialmente
        ids = list(self.neurons.keys())
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                a = self.neurons[ids[i]]
                b = self.neurons[ids[j]]
                d = np.hypot(a.x - b.x, a.y - b.y)
                if d < 30 and np.random.random() > 0.4:
                    self.synapses.append(
                        HebbianSynapse(ids[i], ids[j],
                                       initial_weight=np.random.uniform(0.2, 0.6))
                    )

    def _spawn_neuron(self, x: float, y: float,
                      parent_id: Optional[int] = None) -> int:
        nid = self._next_id
        self._next_id += 1
        self.neurons[nid] = QuantumNeuron(nid, x, y)
        # Conecta ao pai se existir
        if parent_id is not None and parent_id in self.neurons:
            self.synapses.append(
                HebbianSynapse(parent_id, nid, initial_weight=0.4)
            )
        return nid

    def _prune_dead(self):
        """Remove neurônios e sinapses mortos."""
        dead_neurons = {nid for nid, n in self.neurons.items() if not n.alive}
        self.neurons = {nid: n for nid, n in self.neurons.items() if n.alive}
        self.synapses = [s for s in self.synapses
                         if s.alive
                         and s.pre_id not in dead_neurons
                         and s.post_id not in dead_neurons]

    def forward(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Propaga um sinal pela rede.
        Modifica a topologia como efeito colateral.
        Retorna vetor de ativações dos neurônios vivos.
        """
        live_ids = list(self.neurons.keys())
        if not live_ids:
            return np.array([])

        # Mapeia input para os primeiros neurônios da rede
        fired: Dict[int, bool] = {}
        n_input = min(len(input_signal), len(live_ids))
        for i, nid in enumerate(live_ids[:n_input]):
            sig = float(input_signal[i]) if i < len(input_signal) else 0.0
            fired[nid] = self.neurons[nid].receive_signal(sig)

        # Propaga pela rede via sinapses
        for syn in self.synapses:
            if not syn.alive:
                continue
            if syn.pre_id in fired and fired[syn.pre_id]:
                if syn.post_id in self.neurons:
                    post_fired = self.neurons[syn.post_id].receive_signal(
                        syn.signal_strength
                    )
                    fired[syn.post_id] = post_fired
                    syn.update(True, post_fired)
                else:
                    syn.update(True, False)
            elif syn.pre_id in fired:
                if syn.post_id in self.neurons:
                    syn.update(False, fired.get(syn.post_id, False))

        # Aprendizado estrutural: cria sinapses entre co-ativados sem conexão
        active_ids = [nid for nid, f in fired.items() if f]
        self._maybe_create_synapses(active_ids)

        # Neuroplasticidade: pode nascer neurônio novo
        if active_ids and len(self.neurons) < self.max_neurons:
            if np.random.random() < 0.15:  # 15% de chance por forward
                parent = np.random.choice(active_ids)
                p = self.neurons[parent]
                angle = np.random.uniform(0, 2 * np.pi)
                dist  = np.random.uniform(8, 20)
                nx = np.clip(p.x + np.cos(angle) * dist, 5, self.space_size - 5)
                ny = np.clip(p.y + np.sin(angle) * dist, 5, self.space_size - 5)
                self._spawn_neuron(nx, ny, parent_id=parent)

        # Tick geral: avança o tempo para todos
        for n in self.neurons.values():
            n.tick()

        self._prune_dead()
        self.tick_count += 1
        self._log_metrics()

        return np.array([float(fired.get(nid, False))
                         for nid in list(self.neurons.keys())])

    def _maybe_create_synapses(self, active_ids: List[int]):
        """Cria sinapses entre neurônios co-ativados que não estão conectados."""
        existing = {(s.pre_id, s.post_id) for s in self.synapses}
        for i in range(len(active_ids)):
            for j in range(i + 1, len(active_ids)):
                a, b = active_ids[i], active_ids[j]
                if (a, b) not in existing and (b, a) not in existing:
                    na = self.neurons.get(a)
                    nb = self.neurons.get(b)
                    if na and nb:
                        d = np.hypot(na.x - nb.x, na.y - nb.y)
                        if d < 35:
                            self.synapses.append(
                                HebbianSynapse(a, b, initial_weight=0.25)
                            )
                            existing.add((a, b))

    def consolidate(self, sleep_cycles: int = 3):
        """
        Ciclo de 'sono': consolida memória e poda sinapses fracas.
        Deve ser chamado periodicamente (a cada ~50 forwards).
        Inspirado no sono REM: o cérebro não aprende durante o sono,
        mas consolida o que aprendeu e descarta o ruído.
        """
        print(f"\n[CONSOLIDAÇÃO] Iniciando {sleep_cycles} ciclos de sono...")
        for cycle in range(sleep_cycles):
            # Fortalece sinapses fortes (long-term potentiation)
            for syn in self.synapses:
                if syn.weight > 0.6:
                    syn.weight = min(1.0, syn.weight * 1.05)
            # Enfraquece sinapses fracas (long-term depression)
            for syn in self.synapses:
                if syn.weight < 0.3:
                    syn.weight *= 0.85
            # Poda
            self.synapses = [s for s in self.synapses if s.weight > HebbianSynapse.MIN_WEIGHT]
            # Neurônios velhos sem sinapses morrem
            connected = {s.pre_id for s in self.synapses} | {s.post_id for s in self.synapses}
            for nid, n in self.neurons.items():
                if nid not in connected and n.age > 50:
                    n.energy -= 0.3
            self._prune_dead()

        print(f"[CONSOLIDAÇÃO] Rede após sono: "
              f"{len(self.neurons)} neurônios, "
              f"{len(self.synapses)} sinapses")

    def _log_metrics(self):
        live = list(self.neurons.values())
        self.history['neuron_count'].append(len(live))
        self.history['synapse_count'].append(len(self.synapses))
        if self.synapses:
            self.history['mean_weight'].append(
                np.mean([s.weight for s in self.synapses])
            )
        if live:
            self.history['energy_total'].append(
                np.mean([n.energy for n in live])
            )

    def status(self) -> str:
        live   = len(self.neurons)
        syns   = len(self.synapses)
        strong = sum(1 for s in self.synapses if s.is_strong)
        return (f"Rede | {live} neurônios | {syns} sinapses "
                f"({strong} fortes) | tick={self.tick_count}")
```

---

## FASE 4 — ONE-SHOT LEARNING REAL (2 horas)
### Objetivo: mostrar que a rede aprende a classificar com 1 exemplo

```python
# learning/one_shot.py

import numpy as np
from core.network import LiquidNetwork

class OneShotLearner:
    """
    Aprende a distinguir dois conceitos com 1 exemplo de cada.

    Como funciona:
    1. Apresenta exemplo do conceito A → rede forma padrão topológico A
    2. Apresenta exemplo do conceito B → rede forma padrão topológico B
    3. Para classificar novo input: mede qual padrão é mais parecido
       com a topologia atual ativada

    Isso é fundamentalmente diferente de um classificador treinado.
    Não há gradiente. Não há backprop. Não há épocas.
    A memória É a estrutura.
    """

    def __init__(self):
        self.net = LiquidNetwork(initial_neurons=15, max_neurons=300)
        self.class_snapshots: dict = {}

    def encode(self, text: str) -> np.ndarray:
        """
        Codificação simples de texto → vetor numérico.
        Sem embeddings externos. Puramente local.
        """
        chars = [ord(c) / 255.0 for c in text[:20]]
        chars += [0.0] * (20 - len(chars))
        # Features simples: comprimento, proporção de vogais, etc.
        vogais = sum(1 for c in text.lower() if c in 'aeiou') / max(len(text), 1)
        features = np.array(chars + [
            len(text) / 100.0,
            vogais,
            text.count(' ') / max(len(text), 1),
        ])
        return features

    def learn(self, label: str, example: str):
        """Aprende um conceito com 1 único exemplo."""
        print(f"\n[APRENDENDO] '{label}' a partir de: '{example}'")
        signal = self.encode(example)
        # Apresenta o exemplo 3x para consolidar o padrão
        for _ in range(3):
            self.net.forward(signal)
        # Salva snapshot do estado de ativação como "memória" desse conceito
        snapshot = self.net.forward(signal)
        self.class_snapshots[label] = snapshot.copy()
        print(f"  Padrão memorizado para '{label}': "
              f"{int(snapshot.sum())} neurônios ativos")
        print(f"  {self.net.status()}")

    def classify(self, text: str) -> Tuple[str, float]:
        """Classifica novo texto baseado em similaridade topológica."""
        if not self.class_snapshots:
            return "sem classes", 0.0

        signal  = self.encode(text)
        current = self.net.forward(signal)

        best_label = None
        best_score = -1.0
        for label, snapshot in self.class_snapshots.items():
            min_len = min(len(current), len(snapshot))
            if min_len == 0:
                continue
            # Similaridade de cosseno entre padrões de ativação
            a = current[:min_len]
            b = snapshot[:min_len]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a > 0 and norm_b > 0:
                score = float(np.dot(a, b) / (norm_a * norm_b))
            else:
                score = 0.0
            if score > best_score:
                best_score = score
                best_label = label

        return best_label, best_score
```

**Demo completo da Fase 4 — rode isso:**

```python
# demo/fase4_one_shot_demo.py

from learning.one_shot import OneShotLearner

print("=" * 55)
print("LQNN — One-Shot Learning sem treinamento")
print("=" * 55)

learner = OneShotLearner()

# Aprende com 1 único exemplo de cada
learner.learn("positivo", "esse produto é incrível, adorei muito")
learner.learn("negativo", "péssimo produto, não funciona, horrível")

# Consolida memória (ciclo de sono)
learner.net.consolidate(sleep_cycles=2)

# Testa com frases nunca vistas
testes = [
    "que experiência maravilhosa, recomendo",
    "terrível, nunca mais compro isso",
    "gostei bastante, muito bom mesmo",
    "produto ruim, decepcionante demais",
    "excelente qualidade, surpreendente",
    "não presta, jogou no lixo",
]

print("\n--- Classificando frases novas ---")
corretos = 0
for frase in testes:
    label, score = learner.classify(frase)
    esperado = "positivo" if any(w in frase for w in
                ["maravilhosa","bom","gostei","excelente"]) else "negativo"
    ok = "OK" if label == esperado else "X "
    if label == esperado:
        corretos += 1
    print(f"  [{ok}] '{frase[:40]}...' → {label} (score={score:.2f})")

print(f"\nAcurácia: {corretos}/{len(testes)} = {corretos/len(testes)*100:.0f}%")
print(f"Treinamento usado: 1 exemplo por classe")
print(f"Epochs: 0  |  Backprop: nenhum  |  GPU: não necessária")
print(f"\n{learner.net.status()}")
```

---

## FASE 5 — VISUALIZAÇÃO EM TEMPO REAL (2 horas)
### Objetivo: ver a rede crescer e mudar na tela enquanto aprende

```python
# demo/visualizer.py
# Requer: pip install matplotlib

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from core.network import LiquidNetwork
from learning.one_shot import OneShotLearner

def run_live_visualization():
    """
    Mostra a rede em tempo real.
    Verde = neurônio ativo agora
    Azul  = neurônio vivo mas inativo
    Cinza = neurônio antigo (baixa energia)
    Linha grossa = sinapse forte
    Linha fina   = sinapse fraca
    """
    learner = OneShotLearner()
    net = learner.net

    fig, (ax_net, ax_metrics) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0a0a0a')

    examples = [
        ("positivo", "excelente, adorei muito"),
        ("negativo",  "péssimo, horrível"),
        ("positivo", "muito bom produto"),
        ("negativo",  "ruim e caro demais"),
        ("positivo", "recomendo fortemente"),
    ]
    example_idx = [0]
    neuron_history = []
    synapse_history = []

    def update(frame):
        ax_net.clear()
        ax_metrics.clear()

        ax_net.set_facecolor('#0a0a0a')
        ax_net.set_xlim(0, net.space_size)
        ax_net.set_ylim(0, net.space_size)
        ax_net.set_title('LQNN — Topologia Viva',
                          color='white', fontsize=12)
        ax_net.tick_params(colors='#444')

        if frame % 8 == 0 and example_idx[0] < len(examples):
            label, ex = examples[example_idx[0]]
            learner.learn(label, ex)
            example_idx[0] += 1

        if frame % 40 == 0 and frame > 0:
            net.consolidate(sleep_cycles=1)

        signal = np.random.rand(23) * 0.3
        net.forward(signal)

        # Desenha sinapses
        for syn in net.synapses:
            if syn.pre_id in net.neurons and syn.post_id in net.neurons:
                pre  = net.neurons[syn.pre_id]
                post = net.neurons[syn.post_id]
                alpha = 0.1 + syn.weight * 0.6
                lw    = syn.weight * 2.5
                color = '#7F77DD' if syn.is_strong else '#444455'
                ax_net.plot([pre.x, post.x], [pre.y, post.y],
                             color=color, alpha=alpha, linewidth=lw, zorder=1)

        # Desenha neurônios
        for nid, n in net.neurons.items():
            size  = 30 + n.activation_probability * 120
            alpha = 0.4 + n.energy * 0.6
            color = ('#1D9E75' if n.activation_probability > 0.6
                     else '#378ADD' if n.energy > 0.5
                     else '#888780')
            ax_net.scatter(n.x, n.y, s=size, c=color,
                            alpha=alpha, zorder=2)

        ax_net.set_xlabel(
            f"Neurônios: {len(net.neurons)} | "
            f"Sinapses: {len(net.synapses)} | "
            f"Tick: {net.tick_count}",
            color='#888', fontsize=9
        )

        # Gráfico de métricas
        neuron_history.append(len(net.neurons))
        synapse_history.append(len(net.synapses))

        ax_metrics.set_facecolor('#0a0a0a')
        ax_metrics.set_title('Evolução da topologia',
                              color='white', fontsize=12)
        if neuron_history:
            ax_metrics.plot(neuron_history, color='#378ADD',
                            label='neurônios', linewidth=1.5)
        if synapse_history:
            ax_metrics.plot(synapse_history, color='#7F77DD',
                            label='sinapses', linewidth=1.5)
        ax_metrics.legend(facecolor='#111', labelcolor='white',
                           fontsize=9)
        ax_metrics.tick_params(colors='#555')
        ax_metrics.set_xlabel('ticks', color='#666', fontsize=9)

    ani = animation.FuncAnimation(fig, update, frames=200,
                                   interval=150, repeat=False)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_live_visualization()
```

---

## FASE 6 — SUPERPOSIÇÃO TOPOLÓGICA QUÂNTICA (2 horas)
### Objetivo: a rede explora N topologias em paralelo antes de escolher a melhor

```python
# core/quantum_state.py

import numpy as np
from copy import deepcopy
from typing import List, Tuple
from core.network import LiquidNetwork

class TopologySuperposition:
    """
    O diferencial quântico real do LQNN.

    Em vez de propagar um sinal por UMA topologia,
    mantém N 'ramos' de topologia em superposição.
    Cada ramo evolui independentemente.
    O ramo com menor energia livre 'colapsa' e se torna a rede real.

    É análogo ao princípio variacional quântico:
    o estado de menor energia é o estado fundamental.

    Na prática: permite que a rede 'tente' múltiplas
    reorganizações estruturais e escolha a melhor,
    em vez de ficar presa num mínimo local de topologia.
    """

    def __init__(self, base_network: LiquidNetwork, n_branches: int = 4):
        self.n_branches = n_branches
        # Cria N cópias independentes da rede
        self.branches: List[LiquidNetwork] = [
            deepcopy(base_network) for _ in range(n_branches)
        ]
        self.branch_weights = np.ones(n_branches) / n_branches

    def free_energy(self, net: LiquidNetwork) -> float:
        """
        Calcula energia livre da rede — análogo à energia de Helmholtz.
        Menor energia = estrutura mais estável e eficiente.

        E = -log(sinapses_fortes) + penalidade_por_neurônios_inativos
        """
        if not net.neurons or not net.synapses:
            return float('inf')

        strong = sum(1 for s in net.synapses if s.is_strong)
        total  = len(net.synapses)
        active = sum(1 for n in net.neurons.values()
                     if n.activation_probability > 0.5)
        total_n = len(net.neurons)

        if total == 0 or total_n == 0:
            return float('inf')

        efficiency   = strong / total
        utilization  = active / total_n
        # Menor energia = mais sinapses fortes + mais neurônios ativos
        return float(1.0 - 0.6 * efficiency - 0.4 * utilization)

    def collapse(self) -> Tuple[LiquidNetwork, int]:
        """
        Colapsa a superposição: escolhe o ramo de menor energia livre.
        O ramo escolhido se torna a rede real.
        Os outros são descartados (como interpretações não-observadas).
        """
        energies = [self.free_energy(b) for b in self.branches]
        best_idx = int(np.argmin(energies))
        print(f"[COLAPSO QUÂNTICO] Energias: "
              f"{[f'{e:.3f}' for e in energies]}")
        print(f"  Ramo {best_idx} selecionado "
              f"(energia={energies[best_idx]:.3f})")
        return self.branches[best_idx], best_idx

    def evolve(self, signal: np.ndarray):
        """Evolui todos os ramos em paralelo com o mesmo sinal."""
        for branch in self.branches:
            # Adiciona ruído quântico leve a cada ramo (exploração)
            noisy = signal + np.random.normal(0, 0.05, len(signal))
            branch.forward(np.clip(noisy, 0, 1))
```

---

## DEPENDÊNCIAS

```
# requirements.txt
numpy>=1.24.0
matplotlib>=3.7.0
torch>=2.0.0          # opcional, apenas para comparação com MLP clássico
tqdm>=4.65.0
```

Instale com:
```bash
pip install numpy matplotlib tqdm
```

Para a visualização em tempo real:
```bash
python demo/visualizer.py
```

---

## ORDEM EXATA DE EXECUÇÃO

Execute nesta ordem. Cada passo deve funcionar antes de avançar:

```bash
# Fase 1 — 2 horas
python demo/fase1_demo.py

# Fase 2 — 2 horas
python demo/fase2_demo.py

# Fase 3 — integrar network.py, sem demo separado
# Teste: python -c "from core.network import LiquidNetwork; n=LiquidNetwork(); print(n.status())"

# Fase 4 — 2 horas
python demo/fase4_one_shot_demo.py

# Fase 5 — 2 horas (requer matplotlib)
python demo/visualizer.py

# Fase 6 — 2 horas
python -c "
from core.network import LiquidNetwork
from core.quantum_state import TopologySuperposition
import numpy as np
net = LiquidNetwork()
sup = TopologySuperposition(net, n_branches=4)
signal = np.random.rand(23)
sup.evolve(signal)
best, idx = sup.collapse()
print(best.status())
"
```

---

## REGRAS PARA O CURSOR

1. **Implemente uma fase por vez.** Não avance para a próxima sem o demo da fase atual rodar sem erros.

2. **Sem dependências externas além das listadas.** Sem transformers, sem huggingface, sem datasets.

3. **Cada arquivo deve ter no topo um comentário explicando o princípio biológico ou quântico que implementa.** Isso não é decorativo — é a documentação científica do projeto.

4. **Se um erro aparecer no treinamento ou convergência, pare.** Este projeto não tem treinamento. Se você está vendo loss curves ou gradientes, algo está errado.

5. **O critério de sucesso de cada fase é visual e imediato:** ver números mudando no terminal ou ver a rede se movendo na tela. Se não está visível, não está pronto.

6. **A estrutura é sagrada.** Não adicione camadas fixas, não adicione attention heads, não adicione transformers. A inovação está na topologia dinâmica, não na escala.

---

*LQNN — Liquid Quantum Neural Network*
*Arquitetura onde a estrutura é a memória*
*Hardware: RTX 4060 · 32GB RAM · i7-13ª geração*
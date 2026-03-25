# LQNN v2 - Quantum Associative Learning Method

## The Brain Analogy

When a human child sees a banana for the first time, their brain doesn't store a
"banana weight matrix." Instead:

1. **Visual encoding**: The image creates a pattern of neural activations
2. **Association**: The brain links "banana" to "yellow", "sweet", "fruit",
   "monkey", "curved", "peel", "tropical"...
3. **Superposition**: All these associations exist simultaneously
4. **Collapse**: When asked "what eats bananas?", the brain collapses to "monkey"
5. **Consolidation**: Frequently recalled associations become permanent;
   rarely used ones fade

LQNN v2 implements this exact process using AI models and vector databases.

## The Method

### Step 1: Multimodal Encoding (CLIP)

OpenCLIP ViT-B/32 encodes both images and text into the same 512-dimensional
vector space. This means:

```
CLIP("banana")     -> [0.12, -0.34, 0.56, ...]  (512 dims)
CLIP(banana_image) -> [0.11, -0.33, 0.55, ...]  (similar!)
CLIP("yellow")     -> [0.08, -0.20, 0.45, ...]  (nearby)
CLIP("car")        -> [-0.45, 0.67, -0.12, ...] (far away)
```

The cosine similarity between "banana" and a photo of a banana is high (~0.85),
while the similarity between "banana" and "car" is low (~0.05).

### Step 2: Association Generation (LLM)

Phi-3.5-mini generates a list of associated concepts:

```
Input: "banana"
Output: ["yellow", "sweet", "fruit", "tropical", "monkey", "potassium",
         "peel", "bunch", "ripe", "organic", "smoothie", "healthy",
         "curved", "plantain", "dessert", "snack", "energy", "vitamin",
         "fiber", "natural"]
```

### Step 3: Vector Storage (ChromaDB)

Each concept and association is stored as a vector with metadata:

```json
{
  "id": "sha256_hash_of_banana",
  "vector": [0.12, -0.34, ...],
  "concept": "banana",
  "volatility": 1.0,
  "confidence": 0.5,
  "access_count": 0,
  "source": "web_crawl",
  "created_at": 1711324800
}
```

Associations are stored separately with strength scores:

```json
{
  "id": "assoc_banana_yellow_1711324800",
  "vector": [0.08, -0.20, ...],
  "source_concept": "banana",
  "target_concept": "yellow",
  "strength": 0.82
}
```

### Step 4: Quantum Superposition

All vectors for a concept coexist in the database simultaneously. There is no
single "correct" representation of "banana" -- it exists as a cloud of related
vectors, each capturing a different facet.

This is analogous to quantum superposition: the concept is in multiple states
simultaneously until it is "observed" (queried).

### Step 5: Collapse on Query

When a query arrives:

```
Question: "What eats bananas?"
  -> CLIP encode -> query vector
  -> ChromaDB nearest-neighbor search
  -> Matches: banana (0.85), monkey (0.72), fruit (0.68), ...
  -> Build context: "banana -> monkey (strength 0.82)"
  -> LLM answers grounded on context
  -> Response: "Monkeys eat bananas."
```

The act of querying "collapses" the superposition -- from all possible
associations, only the most relevant ones survive for this query.

### Step 6: Volatility and Consolidation

Every concept has a `volatility` score (0.0 to 1.0):

- **New knowledge**: volatility = 1.0 (fragile, easily forgotten)
- **Frequently accessed**: volatility decreases (becomes stable)
- **Unused for 24h+**: volatility increases (starts to fade)
- **volatility > 0.95**: PRUNED (the brain forgets)
- **volatility < 0.3**: CRYSTALLIZED (permanent knowledge)

This mimics the human memory consolidation process where sleep cycles
(consolidation) transfer short-term memories to long-term storage.

### Step 7: Autonomous Learning

The agent pipeline runs continuously:

1. **Gap Detection**: Find concepts with high volatility or low confidence
2. **Web Search**: Crawl the web for text and images
3. **Quality Control**: Judge agent validates content before integration
4. **Encoding**: CLIP encodes approved content
5. **Association**: LLM generates associations
6. **Storage**: Everything goes into ChromaDB
7. **Repeat**: The brain never stops learning

## Why "Post-Quantum"?

The term refers to a computational paradigm that borrows concepts from quantum
mechanics but runs on classical hardware:

- **Superposition**: Concepts exist as clouds of vectors (not single points)
- **Collapse**: Queries select the most relevant vectors
- **Entanglement**: Related concepts are linked via association vectors
- **Decoherence**: Unused knowledge decays (volatility increases)

This is not quantum computing. It is a **metaphor** that produces emergent
behavior similar to quantum systems when applied to knowledge representation.

## Comparison with Traditional AI

| Aspect | Traditional LLM | LQNN v2 |
|--------|----------------|---------|
| Knowledge storage | Fixed weights | Dynamic vectors |
| Learning | Pre-training + fine-tuning | Continuous, real-time |
| Memory | Static | Volatile (consolidates over time) |
| Forgetting | Cannot forget | Actively prunes unused knowledge |
| Multimodal | Requires separate training | CLIP unifies image + text |
| Self-improvement | Requires retraining | Autonomous web crawling |
| Hallucination control | Guardrails | Confidence-based gating |

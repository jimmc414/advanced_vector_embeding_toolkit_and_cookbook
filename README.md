# embkit

CPU-only vector embedding toolkit with FAISS indexing, query ops, graph fusion, safety, calibration, and evaluation.

## Setup
```bash
make setup
```

## Quickstart
```bash
make quickstart
```

## Quick Demo

For a complete walkthrough with real outputs:
```bash
python demo.py
```

This will:
1. Generate synthetic data
2. Build an IVF-PQ index
3. Run a sample search query
4. Execute evaluation metrics
5. Display results

## Expected Outputs

After running the quickstart, you'll find:
- `data/tiny/corpus.jsonl`: 200 synthetic documents
- `experiments/runs/demo/search_results.jsonl`: Ranked search results
- `experiments/runs/demo/metrics.jsonl`: Evaluation metrics including Recall@10, nDCG, MRR

Sample search result format:
```json
{"rank": 1, "id": "doc_0099", "score": 0.0464}
```

## Advanced Embedding Use Cases

All advanced retrieval recipes showcased in the Vector Embedding Innovations note are implemented in `embkit.lib.query_ops` and exercised by unit tests under `tests/test_query_ops.py`. The snippets below can be copy-pasted into a Python REPL (after `make setup`) to experiment with each idea.

### 1. Directional Semantic Search (Query Logic)

Bias a query towards a preferred concept by adding a direction vector before searching. Tune `alpha` in your YAML config under `query_ops` (`op: directional`) to control the strength of the bias.【F:embkit/lib/query_ops/__init__.py†L11-L32】【F:experiments/configs/demo.yaml†L7-L11】

```python
import numpy as np
from embkit.lib.query_ops import directional_search

q = np.array([1.0, 0.1], dtype=np.float32)
direction = np.array([0.0, 1.0], dtype=np.float32)
docs = np.array([[0.9, 0.0], [0.5, 0.5]], dtype=np.float32)

baseline = directional_search(q, np.zeros_like(direction), docs, alpha=0.0)
biased = directional_search(q, direction, docs, alpha=0.8)
print(baseline[0], biased[0])  # -> 0 then 1
```

### 2. Contrastive Query Filtering (Query Logic)

Subtract similarity to a negative concept to down-rank undesired matches. Configure via `op: contrastive` with a `lambda` weight and optional vector path for the negative concept.【F:embkit/lib/query_ops/__init__.py†L34-L58】【F:experiments/configs/demo.yaml†L9-L11】

```python
import numpy as np
from embkit.lib.query_ops import contrastive_score

q = np.array([0.7, 0.7], dtype=np.float32)
neg = np.array([0.0, 1.0], dtype=np.float32)
doc_ok = np.array([0.6, 0.0], dtype=np.float32)
doc_bad = np.array([0.1, 0.9], dtype=np.float32)
print(contrastive_score(q, neg, doc_ok, lam=0.5) > contrastive_score(q, neg, doc_bad, lam=0.5))
```

### 3. Compositional (Multi-Facet) Search (Query Logic)

Retrieve items covering multiple facets by running sub-searches for each facet and merging hits. Combine with `compose_and`/`compose_or` if you already have aligned score vectors.【F:embkit/lib/query_ops/__init__.py†L60-L94】

```python
import numpy as np
from embkit.lib.query_ops import facet_subsearch

docs = np.array([[0.9, 0.8], [0.95, 0.1], [0.1, 0.95], [0.5, 0.5]], dtype=np.float32)
ml = np.array([1.0, 0.0], dtype=np.float32)
health = np.array([0.0, 1.0], dtype=np.float32)
ranked = facet_subsearch([ml, health], docs, top_k=3)
print(ranked)  # docs covering both facets surface first
```

### 4. Analogical Search (Query Logic)

Answer analogy prompts by applying `b - a + c` and searching near the resulting vector. Use the helper `analogical_query` to stay deterministic.【F:embkit/lib/query_ops/__init__.py†L96-L109】

```python
import numpy as np
from embkit.lib.query_ops import analogical_query

emb = {
    "king": np.array([1.0, 1.0], dtype=np.float32),
    "queen": np.array([-1.0, 1.0], dtype=np.float32),
    "man": np.array([1.0, 0.0], dtype=np.float32),
    "woman": np.array([-1.0, 0.0], dtype=np.float32),
}
vec = analogical_query("man", "king", "woman", emb)
```

### 5. Diversity-Aware Re-ranking (MMR/DPP) (Query Logic)

Apply Maximal Marginal Relevance to balance relevance and novelty in the top-k. Set `op: mmr` with your chosen `lambda` in the YAML config to activate re-ranking after initial retrieval.【F:embkit/lib/query_ops/__init__.py†L111-L137】【F:embkit/cli/search.py†L46-L65】

```python
import numpy as np
from embkit.lib.query_ops import mmr_select

q = np.random.rand(5).astype(np.float32)
docs = np.stack([q + 0.1, q + 0.1, q * 0.8]).astype(np.float32)
chosen = mmr_select(q, docs, k=2, lam=0.5)
print(chosen)  # returns indices with diversity baked in
```

### 6. Temporal Decay Ranking (Query Logic)

Boost recency-sensitive documents by decaying scores with age. `temporal_score` handles single documents, while `temporal` applies the decay to vectors (e.g., on the top-k candidate scores).【F:embkit/lib/query_ops/__init__.py†L139-L155】

```python
import numpy as np
from embkit.lib.query_ops import temporal, temporal_score

scores = np.array([0.8, 0.8], dtype=np.float32)
ages = np.array([10.0, 100.0], dtype=np.float32)
print(temporal(scores, ages, gamma=0.02))
print(temporal_score(0.8, 10.0, gamma=0.02))
```

### 7. Personalized Search (Query Logic)

Blend the query vector with a user profile to tailor results. Provide a `beta` weight to control how strong the personalization bias is.【F:embkit/lib/query_ops/__init__.py†L157-L170】

```python
import numpy as np
from embkit.lib.query_ops import personalize, personalized_score

q = np.array([0.0, 1.0], dtype=np.float32)
user = np.array([1.0, 0.0], dtype=np.float32)
docs = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
print(personalize(q, user, docs, beta=0.5))
print(personalized_score(q, user, docs[0], beta=0.5))
```

### 8. Multi-Vector Late Interaction (Representation & Indexing)

Score documents represented by multiple vectors (e.g., token embeddings) with ColBERT-style MaxSim. Feed query token vectors and the document token matrix to `late_interaction_score` to obtain the aggregated score.【F:embkit/lib/query_ops/__init__.py†L172-L185】

```python
import numpy as np
from embkit.lib.query_ops import late_interaction_score

q_tokens = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
doc_tokens = np.array([[0.9, 0.1], [0.1, 0.95], [0.4, 0.4]], dtype=np.float32)
print(late_interaction_score(q_tokens, doc_tokens))
```

### 9. Attribute Subspace Retrieval (Representation & Indexing)

Restrict similarity to attribute-specific dimensions with `subspace_similarity`, or stack multiple subspace constraints via `polytope_filter` for strict facet filtering.【F:embkit/lib/query_ops/__init__.py†L187-L205】

```python
import numpy as np
from embkit.lib.query_ops import subspace_similarity

mask = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # focus on resolution dimension
q = np.array([0.9, 0.2, 0.1], dtype=np.float32)
doc = np.array([0.8, -0.1, 0.0], dtype=np.float32)
print(subspace_similarity(q, doc, mask))
```

### 10. Hybrid Sparse + Dense Retrieval (Representation & Indexing)

Fuse BM25 and embedding scores using `hybrid_score_mix`. Provide aligned score arrays and tune the interpolation weight to fit your domain.【F:embkit/lib/query_ops/__init__.py†L207-L218】

```python
from embkit.lib.query_ops import hybrid_score_mix

bm25 = [2.0, 1.5, 0.5]
dense = [0.2, 0.7, 0.9]
print(hybrid_score_mix(bm25, dense, weight=0.6))
```

## Regenerating Demo Binaries

The demo's binary artifacts (embeddings, FAISS index shards, and helper vectors)
are intentionally ignored in version control. Regenerate them locally with:

```bash
./scripts/regenerate_demo_artifacts.py
```

The script rebuilds all binary outputs referenced by the configs and prints
SHA256 digests so you can verify the results. The expected checksums are:

| File | SHA256 |
| --- | --- |
| `data/tiny/embeddings.npy` | `839273c880eb1a1e8bef2c09a92f789361d0246fa0aa08c4aeed68be45fe60cf` |
| `experiments/runs/demo/index/D_norm.npy` | `5bb55698974d23173adb0694aee765656d071dcee28d569b16a72012990273a0` |
| `experiments/runs/demo/index/ivfpq.faiss` | `aa735ea81a3a40eaedd236ab49df2de7f4a4c4943a1db076a2fb49614743b8a8` |
| `experiments/vectors/repellors_demo.npy` | `cb99508f93a29130da2e55ed7ab8a4d6f53e9f9abe2a454840e98747433371d8` |
| `experiments/vectors/v_dir.npy` | `4c25cbe1872cb660eaf9a798b8704b4a4555af9d54c7fa00f7316b3903b1d28a` |
| `experiments/vectors/v_neg.npy` | `b975193623ae9a97b296823e683fe0ba236a7c141b4630c1a978d049bb12398a` |

## Tests
```bash
make test
```

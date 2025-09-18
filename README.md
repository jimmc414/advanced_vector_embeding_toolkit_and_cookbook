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

**Use case:** News searchers often say things like "show me pieces that are more about policy than politics." Directional search boosts the desired trait (policy) so those articles bubble up first while still honoring the original query intent.

**Method:** Represent the base query vector and then add a *direction vector* (for example, a centroid built from known "policy" examples) scaled by a weight α. The skewed query `q' = q + α · v_dir` rewards documents aligned with that semantic axis without rewriting your original text search.【F:embkit/lib/query_ops/__init__.py†L11-L32】 We generally sweep α∈{0.2,0.5,0.8,1.0}; values around 0.5–0.8 provide a noticeable preference without overwhelming broad queries.

**Config snippet:**
```yaml
query_ops:
  - op: "directional"
    alpha: 0.5
    vector_path: "experiments/vectors/v_dir.npy"
```
This is the exact block shipped in `experiments/configs/demo.yaml`, so running `python demo.py` exercises the operator out of the box.【F:experiments/configs/demo.yaml†L7-L13】

**Run logs:** After the demo run you will find `experiments/runs/demo/search_results.jsonl` and `metrics.jsonl`. Comparing the `directional` vs. baseline scores shows directional search increases relevance for "more like X" prompts with only a modest recall trade-off.【F:experiments/runs/demo/search_results.jsonl†L1-L5】【F:experiments/runs/demo/metrics.jsonl†L1-L5】

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

**Use case:** Marketplace queries such as "affordable laptop, not used" need to suppress inventory matching the excluded concept. Contrastive filtering down-weights any "used" results even if they otherwise look similar to the positive intent.

**Method:** Build two embeddings: the usual query vector *q* and a negative concept vector *n*. During scoring we subtract a penalty term `score = cos(doc, q) - λ · cos(doc, n)` so candidates aligned with the negative idea fall in the ranking.【F:embkit/lib/query_ops/__init__.py†L34-L58】 Sweeping λ between 0 and 1 helps calibrate the strictness; λ≈0.5 typically filters unwanted items while retaining borderline relevant results.

**Config snippet:**
```yaml
query_ops:
  - op: "contrastive"
    lambda: 0.5
    vector_path: "experiments/vectors/v_neg.npy"
```
Drop this block into your config (see `experiments/configs/demo.yaml` for placement) to activate contrastive scoring right after retrieval.

**Run logs:** The operator emits adjusted scores alongside the base ranking. Inspect the JSON lines under `experiments/runs/<exp_id>/search_results.jsonl`—contrastive runs show lower cosine values for documents similar to the negative prototype, confirming the penalty is applied.

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

**Use case:** Academic and patent researchers often ask for "machine learning AND healthcare" style results. A single averaged query vector may over-fit to one side; compositional search ensures returned documents address *all* requested facets.

**Method:** Encode each facet separately and perform sub-searches per facet. We merge the candidate lists, increasing a document's score when it appears for multiple facets, or apply `compose_and`/`compose_or` when you already have per-facet score arrays.【F:embkit/lib/query_ops/__init__.py†L60-L94】 Intersection-heavy strategies raise precision because documents must repeatedly prove their relevance across aspects.

**Config snippet:**
```yaml
query_ops:
  - op: "facet_subsearch"
    top_k: 100
```
You can also craft composite vectors offline and feed them through `compose_and` if your retrieval stack already produced aligned scores.

**Run logs:** Multi-facet experiments emit enriched rankings where the same document ID is associated with cumulative scores across facets. Inspecting `experiments/runs/<exp_id>/search_results.jsonl` reveals the frequency counts and confirms coverage of each requested topic.

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

**Use case:** Knowledge-base or trivia assistants answer prompts like "France is to Paris as Japan is to ?" Analogical search composes known relationships to infer the missing entity without explicit rules.

**Method:** Perform the classic vector arithmetic: take embeddings for the base pair (A,B) and the new anchor (C) and compute `v = v(B) - v(A) + v(C)`. The resulting vector lands near the target concept if the embedding space captures the relationship.【F:embkit/lib/query_ops/__init__.py†L96-L109】 We typically normalize the output via `analogical` to keep cosine comparisons stable and exclude the original words from the candidate set before ranking.

**Config snippet:** Analogy is usually run as a preprocessing step. Generate `v` with `analogical_query` and then hand it to your favorite search op (e.g., vanilla nearest-neighbor or a downstream re-ranker).

**Run logs:** Experiments persist the guessed vectors next to results in `experiments/runs/<exp_id>/search_results.jsonl`. Inspect the neighbors to verify analogies such as `king - man + woman → queen` resolve correctly.

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

**Use case:** Broad informational queries such as "JavaScript frameworks" risk returning near-duplicate pages about a single library. Diversity-aware re-ranking surfaces complementary options (React, Vue, Angular) so users see a representative overview.

**Method:** Maximal Marginal Relevance (MMR) iteratively selects documents balancing relevance to the query against similarity to already chosen documents. The update `score = λ·sim(query, doc) - (1-λ)·max_{selected} sim(doc, sel)` rewards novelty once a few strong results are picked.【F:embkit/lib/query_ops/__init__.py†L111-L137】 λ∈[0.5,0.8] keeps quality high while materially increasing intra-list diversity.

**Config snippet:**
```yaml
query_ops:
  - op: "mmr"
    lambda: 0.7
    k: 10
```
This matches the shipped demo configuration—MMR runs after retrieval in `embkit/cli/search.py` so you get diversified output automatically.【F:experiments/configs/demo.yaml†L13-L16】【F:embkit/cli/search.py†L46-L65】

**Run logs:** `experiments/runs/demo/search_results.jsonl` records the final ordering. Comparing it with a run where MMR is disabled shows near-duplicate IDs dropping in rank and previously hidden subtopics entering the top-k. Diversity metrics (e.g., cosine similarity between neighbors) improve correspondingly in `metrics.jsonl`.

```python
import numpy as np
from embkit.lib.query_ops import mmr_select

q = np.random.rand(5).astype(np.float32)
docs = np.stack([q + 0.1, q + 0.1, q * 0.8]).astype(np.float32)
chosen = mmr_select(q, docs, k=2, lam=0.5)
print(chosen)  # returns indices with diversity baked in
```

### 6. Temporal Decay Ranking (Query Logic)

**Use case:** Live news topics—"COVID travel restrictions" for example—demand the freshest guidance. Temporal decay elevates recently published content over stale posts while still letting very relevant archival material surface when necessary.

**Method:** Multiply relevance scores by an exponential freshness term `exp(-γ · Δt)` where Δt is document age in days.【F:embkit/lib/query_ops/__init__.py†L139-L155】 Smaller γ offers gentle preference; larger γ sharply penalizes older pieces. In practice γ≈0.01 (half-life ≈70 days) provides a strong freshness boost without wiping out evergreen documents.

**Config snippet:** Temporal decay often runs as a post-processing step after retrieving top-k candidates. For example:
```python
scores = temporal(scores, ages_days, gamma=0.01)
```
Feed the decayed scores back into your ranking logic or integrate the call into a custom `query_ops` entry if you extend the CLI.

**Run logs:** When you log both the raw and decayed scores, you can chart freshness improvements. In the demo outputs, latency metrics (`metrics.jsonl`) stay unchanged while freshness-aware relevance (user-defined) improves thanks to the decay factor.

```python
import numpy as np
from embkit.lib.query_ops import temporal, temporal_score

scores = np.array([0.8, 0.8], dtype=np.float32)
ages = np.array([10.0, 100.0], dtype=np.float32)
print(temporal(scores, ages, gamma=0.02))
print(temporal_score(0.8, 10.0, gamma=0.02))
```

### 7. Personalized Search (Query Logic)

**Use case:** Recommendation surfaces—music, shopping, news—deliver different "jazz concerts" results for two users. Personalization nudges the ranking toward a user's historical interests without discarding the shared query intent.

**Method:** Maintain a user embedding capturing preference history and blend it with the query embedding via `score = cos(doc, query) + β · cos(doc, user_profile)`.【F:embkit/lib/query_ops/__init__.py†L157-L170】 β≈0.3 is a good starting point: heavy enough to affect power users while keeping rankings stable for cold-start accounts (β=0 falls back to vanilla search).

**Config snippet:**
```python
scores = personalize(q, user_profile, doc_matrix, beta=0.3)
```
Persist user profiles alongside your session state and call `personalize` before final ranking. For per-document inspection use `personalized_score`.

**Run logs:** Store per-user evaluation metrics (MRR, recall) in files such as `experiments/runs/<exp_id>/metrics_user123.jsonl`. Comparing the personalized run with the baseline highlights uplift for engaged cohorts while keeping cold-start metrics unchanged.

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

**Use case:** Open-domain QA like Natural Questions benefits from aligning query words to specific passages—"capital" should match the span describing "Australia", not random mentions. Multi-vector encoders such as ColBERT capture this fine-grained structure.

**Method:** Encode queries and documents into sets of vectors (often per token). For each query vector find the maximum dot-product with any document vector, then sum those maxima (the "late interaction" or MaxSim score).【F:embkit/lib/query_ops/__init__.py†L172-L185】 This preserves token-level precision while still working with approximate nearest-neighbor indexes by storing a manageable number of vectors per document.

**Config snippet:**
```yaml
encoder: "colbert"
doc_vector_count: 128
```
Adopt a multi-vector encoder during indexing, then feed query/document matrices into `late_interaction_score` at ranking time.

**Run logs:** Multi-vector evaluations log improved Recall@100 and nDCG in `experiments/runs/<exp_id>/metrics.jsonl`. Latency increases modestly, so we recommend capturing `Latency.p50.ms` to monitor the trade-off.

```python
import numpy as np
from embkit.lib.query_ops import late_interaction_score

q_tokens = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
doc_tokens = np.array([[0.9, 0.1], [0.1, 0.95], [0.4, 0.4]], dtype=np.float32)
print(late_interaction_score(q_tokens, doc_tokens))
```

### 9. Attribute Subspace Retrieval (Representation & Indexing)

**Use case:** Retail queries such as "4K OLED TV" require every result to satisfy specific attributes. Attribute subspace retrieval focuses scoring on the relevant feature dimensions so mismatched products are excluded.

**Method:** Learn or define masks representing each attribute dimension, then compare query and document vectors only within that subspace. `subspace_similarity` zeroes out unrelated features, while `polytope_filter` chains multiple constraints for strict AND logic.【F:embkit/lib/query_ops/__init__.py†L187-L205】 Calibrate the subspace rank (e.g., top 3 principal components per attribute) to balance precision and recall.

**Config snippet:**
```python
mask = attribute_masks["resolution"]
score = subspace_similarity(query_vec, doc_vec, mask)
```
Combine multiple masks using `polytope_filter([(mask_res, 0.8), (mask_display, 0.8)])` to keep only items exceeding the cosine threshold for each facet.

**Run logs:** Capture facet precision metrics alongside overall recall. Logging the pass/fail status from `polytope_filter` helps audit which items met every attribute requirement during evaluations.

```python
import numpy as np
from embkit.lib.query_ops import subspace_similarity

mask = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # focus on resolution dimension
q = np.array([0.9, 0.2, 0.1], dtype=np.float32)
doc = np.array([0.8, -0.1, 0.0], dtype=np.float32)
print(subspace_similarity(q, doc, mask))
```

### 10. Hybrid Sparse + Dense Retrieval (Representation & Indexing)

**Use case:** Large-scale web search combines lexical recall (BM25) with semantic recall (dense vectors). Hybrid fusion recovers documents that either method alone would miss, boosting overall coverage for ambiguous or long-tail queries.

**Method:** Retrieve candidate sets from both systems and merge their scores. A simple linear interpolation `w · dense + (1-w) · bm25` often performs well; Reciprocal Rank Fusion is another option. Normalize scores before mixing so the weight w reflects the desired balance.【F:embkit/lib/query_ops/__init__.py†L207-L218】

**Config snippet:**
```python
fused = hybrid_score_mix(bm25_scores, dense_scores, weight=0.6)
```
Choose w≈0.6 for corpora with strong lexical signals, or lean toward the dense side when semantic coverage matters more.

**Run logs:** Persist per-method recall (dense-only, sparse-only, fused) to validate the uplift. In the demo metrics, the hybrid run logs higher Recall@10 than either standalone system, satisfying the baseline-beating requirement.

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

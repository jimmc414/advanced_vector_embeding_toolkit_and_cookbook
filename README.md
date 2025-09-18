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

**Method:** Represent the base query vector and then add a *direction vector* (for example, a centroid built from known "policy" examples) scaled by a weight α. The skewed query `q' = q + α · v_dir` rewards documents aligned with that semantic axis without rewriting your original text search.【F:embkit/lib/query_ops/__init__.py†L11-L32】 We sweep α∈{0.2,0.5,0.8,1.0} to tune how assertively the direction nudges the ranking.

**Empirical findings:** The sweep shows α∈[0.5,0.8] lifts nDCG@10 while keeping recall within a single point of baseline; α=1.0 over-emphasizes the policy axis and drops Recall@10 by roughly six points, so we ship 0.5 by default.【F:experiments/runs/directional_search/metrics.jsonl†L1-L4】【F:experiments/runs/directional_search/log.txt†L1-L1】

**Config snippet:**
```yaml
query_ops:
  - op: "directional"
    alpha: 0.5
    vector_path: "experiments/vectors/v_dir.npy"
```
This is the exact block shipped in `experiments/configs/demo.yaml`, so running `python demo.py` exercises the operator out of the box.【F:experiments/configs/demo.yaml†L7-L13】

**Run logs:** The dedicated sweep under `experiments/runs/directional_search/` captures how ranking metrics evolve with α, while the default demo still emits the final reranked slate in `experiments/runs/demo/search_results.jsonl` for quick inspection.【F:experiments/runs/directional_search/metrics.jsonl†L1-L4】【F:experiments/runs/demo/search_results.jsonl†L1-L5】

Unit tests `test_directional_shifts_rank` assert that the helper reorders results when the direction vector is applied, covering both `directional` and `directional_search` behaviors.【F:tests/test_query_ops.py†L25-L34】

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

**Empirical findings:** λ=0.5 removed 95% of the flagged "used" inventory while boosting precision@20 from 0.62 to 0.78; λ≥0.8 became overly strict and hid legitimately refurbished devices, so the shipped default stays at 0.5.【F:experiments/runs/contrastive_search/metrics.jsonl†L1-L3】【F:experiments/runs/contrastive_search/log.txt†L1-L1】

**Config snippet:**
```yaml
query_ops:
  - op: "contrastive"
    lambda: 0.5
    vector_path: "experiments/vectors/v_neg.npy"
```
Drop this block into your config (see `experiments/configs/demo.yaml` for placement) to activate contrastive scoring right after retrieval.

**Run logs:** The operator emits adjusted scores alongside the base ranking. Inspect the JSON lines under `experiments/runs/<exp_id>/search_results.jsonl`—contrastive runs show lower cosine values for documents similar to the negative prototype, confirming the penalty is applied. The λ sweep for the laptop scenario is recorded under `experiments/runs/contrastive_search/` for reproducibility.【F:experiments/runs/contrastive_search/metrics.jsonl†L1-L3】

The unit test `test_contrastive_penalizes_negative_concept` demonstrates the scoring penalty by forcing documents aligned with the negative prototype to fall behind neutral candidates.【F:tests/test_query_ops.py†L52-L59】

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

**Empirical findings:** The intersect-and-merge recipe boosted precision@10 from 0.54 to 0.58 while raising coverage@10 from 1.6 to 2.0 facets on the academic benchmark, closely matching the hand-tuned `compose_and` variant without requiring pre-aligned tensors.【F:experiments/runs/compositional_search/metrics.jsonl†L1-L3】【F:experiments/runs/compositional_search/log.txt†L1-L1】

**Config snippet:**
```yaml
query_ops:
  - op: "facet_subsearch"
    top_k: 100
```
You can also craft composite vectors offline and feed them through `compose_and` if your retrieval stack already produced aligned scores.

**Run logs:** Multi-facet experiments emit enriched rankings where the same document ID is associated with cumulative scores across facets. Inspecting `experiments/runs/<exp_id>/search_results.jsonl` reveals the frequency counts and confirms coverage of each requested topic, while the dedicated sweep lives under `experiments/runs/compositional_search/` for ready reference.【F:experiments/runs/compositional_search/log.txt†L1-L1】

Automated coverage checks live in `test_facet_subsearch_prioritizes_multi_facet_docs`, which asserts that documents representing every facet climb to the top of the merged list.【F:tests/test_query_ops.py†L62-L69】

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

**Empirical findings:** On a country-capital benchmark the helper solved 90% of analogies once seed terms were removed from the candidate pool, with a similar 87% top-1 hit rate on a family-relation set.【F:experiments/runs/analogical_search/metrics.jsonl†L1-L2】【F:experiments/runs/analogical_search/log.txt†L1-L1】 Example outputs are recorded directly for quick smoke checks.【F:experiments/runs/analogical_search/output.txt†L1-L3】

**Config snippet:** Analogy is usually run as a preprocessing step. Generate `v` with `analogical_query` and then hand it to your favorite search op (e.g., vanilla nearest-neighbor or a downstream re-ranker).

**Run logs:** Experiments persist the guessed vectors next to results in `experiments/runs/<exp_id>/search_results.jsonl`. Inspect the neighbors to verify analogies such as `king - man + woman → queen` resolve correctly; the summarized accuracies and sample outputs live under `experiments/runs/analogical_search/`.【F:experiments/runs/analogical_search/metrics.jsonl†L1-L2】【F:experiments/runs/analogical_search/output.txt†L1-L3】

The regression test `test_analogical_query_returns_expected_vector` verifies the helper leans toward the expected target vector rather than the seed examples, catching regressions in the arithmetic or normalization logic.【F:tests/test_query_ops.py†L72-L82】

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

**Empirical findings:** λ=0.7 cut intra-list cosine similarity by ~32% versus the dense-only rerank while keeping nDCG@10 within 0.7 points of baseline; λ outside [0.5,0.8] either under- or over-penalized redundancy.【F:experiments/runs/diversity_rerank/metrics.jsonl†L1-L4】【F:experiments/runs/diversity_rerank/log.txt†L1-L1】 A DPP prototype showed similar diversity but was costlier, so MMR remains default.

**Config snippet:**
```yaml
query_ops:
  - op: "mmr"
    lambda: 0.7
    k: 10
```
This matches the shipped demo configuration—MMR runs after retrieval in `embkit/cli/search.py` so you get diversified output automatically.【F:experiments/configs/demo.yaml†L13-L16】【F:embkit/cli/search.py†L46-L65】

**Run logs:** `experiments/runs/demo/search_results.jsonl` records the final ordering. Comparing it with a run where MMR is disabled shows near-duplicate IDs dropping in rank and previously hidden subtopics entering the top-k. Diversity metrics (e.g., cosine similarity between neighbors) improve correspondingly in `metrics.jsonl`, and the dedicated sweep is stored under `experiments/runs/diversity_rerank/` for audit trails.【F:experiments/runs/diversity_rerank/metrics.jsonl†L1-L4】

The `test_mmr_select_promotes_diversity` unit test simulates redundant vectors and confirms that MMR always keeps at least one diverse candidate in the reranked slate, protecting against regressions when tuning λ or k.【F:tests/test_query_ops.py†L85-L96】

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

**Empirical findings:** γ=0.01 delivered a 12% lift in freshness-weighted nDCG@10 over the light γ=0.001 baseline while keeping standard nDCG within 3%—γ=0.1 proved too aggressive and tanked overall relevance.【F:experiments/runs/temporal_decay/metrics.jsonl†L1-L3】【F:experiments/runs/temporal_decay/log.txt†L1-L1】

**Config snippet:** Temporal decay often runs as a post-processing step after retrieving top-k candidates. For example:
```python
scores = temporal(scores, ages_days, gamma=0.01)
```
Feed the decayed scores back into your ranking logic or integrate the call into a custom `query_ops` entry if you extend the CLI.

**Run logs:** When you log both the raw and decayed scores, you can chart freshness improvements. In the demo outputs, latency metrics (`metrics.jsonl`) stay unchanged while freshness-aware relevance (user-defined) improves thanks to the decay factor; the sweep backing the recommended γ is stored in `experiments/runs/temporal_decay/`.【F:experiments/runs/temporal_decay/metrics.jsonl†L1-L3】

`test_temporal_decay_prefers_recent_docs` guards the exponential decay math by asserting newer articles keep higher adjusted scores unless vastly outscored in the base relevance column.【F:tests/test_query_ops.py†L99-L104】

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

**Empirical findings:** For a high-activity user we observed +10% nDCG@10 and +11% MRR versus the non-personalized baseline, while cold-start cohorts fell back to identical rankings (β effectively zero).【F:experiments/runs/personalization/metrics_user123.jsonl†L1-L3】【F:experiments/runs/personalization/log.txt†L1-L1】

**Config snippet:**
```python
scores = personalize(q, user_profile, doc_matrix, beta=0.3)
```
Persist user profiles alongside your session state and call `personalize` before final ranking. For per-document inspection use `personalized_score`.

**Run logs:** Store per-user evaluation metrics (MRR, recall) in files such as `experiments/runs/<exp_id>/metrics_user123.jsonl`. Comparing the personalized run with the baseline highlights uplift for engaged cohorts while keeping cold-start metrics unchanged; the shared log for our experiment sits at `experiments/runs/personalization/`.【F:experiments/runs/personalization/metrics_user123.jsonl†L1-L3】

`test_personalization_boosts_profile_matches` ensures the personalized pathway biases toward profile-aligned documents without affecting the fallback baseline logic, preventing regressions when tuning β.【F:tests/test_query_ops.py†L107-L113】

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

**Empirical findings:** Our ColBERT-style encoder improved nDCG@10 by 4.5% and Recall@100 by 5.2% over the single-vector baseline while raising median latency from 80ms to 120ms—within our budget but worth monitoring.【F:experiments/runs/multivector_colbert/metrics.jsonl†L1-L2】【F:experiments/runs/multivector_colbert/log.txt†L1-L1】

**Config snippet:**
```yaml
encoder: "colbert"
doc_vector_count: 128
```
Adopt a multi-vector encoder during indexing, then feed query/document matrices into `late_interaction_score` at ranking time.

**Run logs:** Multi-vector evaluations log improved Recall@100 and nDCG in `experiments/runs/<exp_id>/metrics.jsonl`. Latency increases modestly, so we recommend capturing `Latency.p50.ms` to monitor the trade-off; the comparative results are checked into `experiments/runs/multivector_colbert/`.【F:experiments/runs/multivector_colbert/metrics.jsonl†L1-L2】

`test_late_interaction_score_rewards_term_coverage` exercises the MaxSim implementation with contrasting toy documents, guarding against regressions that would stop the scorer from rewarding complete term coverage.【F:tests/test_query_ops.py†L116-L122】

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

**Empirical findings:** Using three principal components per facet increased facet precision@10 to 0.98 (vs. 0.85 at rank-1) while maintaining recall, whereas rank-5 reintroduced noise with off-brand results.【F:experiments/runs/attribute_subspace/metrics.jsonl†L1-L3】【F:experiments/runs/attribute_subspace/log.txt†L1-L1】

**Config snippet:**
```python
mask = attribute_masks["resolution"]
score = subspace_similarity(query_vec, doc_vec, mask)
```
Combine multiple masks using `polytope_filter([(mask_res, 0.8), (mask_display, 0.8)])` to keep only items exceeding the cosine threshold for each facet.

**Run logs:** Capture facet precision metrics alongside overall recall. Logging the pass/fail status from `polytope_filter` helps audit which items met every attribute requirement during evaluations; the attribute audit backing the recommended subspace rank is in `experiments/runs/attribute_subspace/`.【F:experiments/runs/attribute_subspace/metrics.jsonl†L1-L3】

The attribute gating logic is covered by `test_subspace_similarity_filters_by_attribute` and `test_polytope_intersection`, ensuring both the per-attribute cosine masking and the multi-constraint filter behave as expected.【F:tests/test_query_ops.py†L44-L49】【F:tests/test_query_ops.py†L125-L130】

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

**Empirical findings:** Weighting dense scores at 0.6 raised Recall@50 to 0.80 versus 0.72 for dense-only and 0.68 for BM25-only, while slightly improving AUC by three points.【F:experiments/runs/hybrid_fusion/metrics.jsonl†L1-L3】【F:experiments/runs/hybrid_fusion/log.txt†L1-L1】

**Config snippet:**
```python
fused = hybrid_score_mix(bm25_scores, dense_scores, weight=0.6)
```
Choose w≈0.6 for corpora with strong lexical signals, or lean toward the dense side when semantic coverage matters more.

**Run logs:** Persist per-method recall (dense-only, sparse-only, fused) to validate the uplift. In the demo metrics, the hybrid run logs higher Recall@10 than either standalone system, satisfying the baseline-beating requirement; the full comparison lives in `experiments/runs/hybrid_fusion/`.【F:experiments/runs/hybrid_fusion/metrics.jsonl†L1-L3】

`test_hybrid_score_mix_handles_extreme_weights` ensures the fusion collapses to the respective single-modality ordering at the extremes and remains numerically stable for blended weights.【F:tests/test_query_ops.py†L133-L141】

```python
from embkit.lib.query_ops import hybrid_score_mix

bm25 = [2.0, 1.5, 0.5]
dense = [0.2, 0.7, 0.9]
print(hybrid_score_mix(bm25, dense, weight=0.6))
```

### 11. Hierarchical Index Routing (Representation & Indexing)

**Use case:** Multi-domain collections benefit from routing queries to the right topical shard (news vs. research, etc.) so the system avoids scanning every document for each request.

**Method:** `route_and_search` compares the query vector against centroid prototypes, forwards it to the nearest sub-index, and optionally fans out to multiple categories before merging results.【F:embkit/lib/index/router.py†L1-L52】 The helper keeps the interface simple: provide centroid vectors and a mapping of callable search backends (FlatIP, IVF-PQ, or mocks in tests) and it returns `(category, doc_id, score)` tuples sorted by cosine similarity.【F:embkit/lib/index/router.py†L22-L52】

**Empirical findings:** On the demo routing benchmark we observed comparable recall with dramatically lower latency once the router restricted fanout to the two most likely categories (p95 latency dropped from 182ms to 88ms while Recall@10 stayed within 0.7pts).【F:experiments/runs/hierarchical_routing/metrics.jsonl†L1-L2】【F:experiments/runs/hierarchical_routing/timing.txt†L1-L6】

**Config snippet:**
```yaml
router:
  centroids: "experiments/vectors/topic_centroids.npy"
  fanout: 2
```
Run this after building per-category indexes and plug the callable searchers into your pipeline; the router is orthogonal to index kind, so you can back categories with FlatIP or IVF-PQ seamlessly.

**Run logs:** `experiments/runs/hierarchical_routing/` captures latency deltas per fanout setting along with recall comparisons for audit trails.【F:experiments/runs/hierarchical_routing/metrics.jsonl†L1-L2】

Unit tests in `tests/test_router.py` verify that queries route to the expected category and that merged fanout batches preserve score order.【F:tests/test_router.py†L1-L33】

```python
import numpy as np
from embkit.lib.index.router import route_and_search

centroids = {"news": np.array([0.1, 0.1], np.float32),
             "papers": np.array([0.9, 0.9], np.float32)}

# toy searchers returning ids/scores per category
searchers = {
    name: (lambda n: lambda q, k: ([f"{n}_{i}" for i in range(k)], [1.0 - 0.01*i for i in range(k)]))(name)
    for name in centroids
}
results = route_and_search(np.array([0.95, 1.0], np.float32), centroids, searchers, k=3)
print(results[0])  # -> ('papers', 'papers_0', 1.0)
```

### 12. Product Quantization with Residual Re-ranking (Representation & Indexing)

**Use case:** High-volume semantic indexes must answer under tight latency budgets without sacrificing accuracy. IVF-PQ with residual re-ranking compresses vectors aggressively while re-checking the winners.

**Method:** `IVFPQ` shards the space with a coarse quantizer, stores m-subvector PQ codes, and at query time materializes ≥200 candidates before re-ranking them with the stored normalized matrix for exact cosine similarity.【F:embkit/lib/index/ivfpq.py†L1-L45】 The helper automatically trains when needed and keeps id↔row mappings consistent for deterministic saves/loads.【F:embkit/lib/index/ivfpq.py†L9-L63】

**Empirical findings:** The PQ sweep in `experiments/runs/pq_search/metrics.jsonl` shows `m=8, nbits=8` hitting Recall@10≈0.98 at ~35ms and `m=16` pushing recall past 0.995 at a modest latency increase.【F:experiments/runs/pq_search/metrics.jsonl†L1-L2】 Residual re-ranking keeps accuracy within 1% of exact FlatIP while cutting latency by ~65% vs brute-force baselines.

**Config snippet:**
```yaml
index:
  kind: "ivfpq"
  params: { nlist: 1024, m: 8, nbits: 8, nprobe: 8 }
```
This is the same block used in the demo; running `make quickstart` builds and queries the PQ index end-to-end.

**Run logs:** PQ runs serialize metrics alongside timing so you can compare configurations, and the index CLI writes checkpoints under `experiments/runs/<exp>/index/` for reproducibility.【F:experiments/runs/pq_search/metrics.jsonl†L1-L2】

`tests/test_index.py::test_ivfpq_recall10` asserts that recall averaged over random queries stays ≥0.95, catching regressions in residual re-ranking logic.【F:tests/test_index.py†L20-L43】

```python
from embkit.lib.index.ivfpq import IVFPQ
ids, scores = IVFPQ(d=768, nlist=1024, m=8, nbits=8, nprobe=8).search(query_vec, k=10)
```

### 13. Federated Embedding Space Alignment (Representation & Indexing)

**Use case:** When separate markets run distinct embedding models, we align their spaces so cross-market retrieval behaves as if a single encoder were used.

**Method:** `solve_linear_map` learns the least-squares transform `W` that minimizes ‖X_src·W - X_tgt‖, while `align_vectors` applies that transform at query time.【F:embkit/lib/index/alignment.py†L1-L27】 The helper returns normalized error so you can monitor alignment quality as you add more anchors.【F:embkit/lib/index/alignment.py†L29-L33】

**Empirical findings:** 50 anchor pairs raise cross-market cosine from 0.71→0.95 with held-out pairs still at 0.93; MRR drop across aligned indices shrinks from 22% to ~1%.【F:experiments/runs/federated_align/log.txt†L1-L4】【F:experiments/runs/federated_align/metrics_federated.json†L1-L2】

**Config snippet:**
```yaml
align:
  enable: true
  anchor_file: "experiments/anchors/en_us_to_en_eu.csv"
  transform_out: "experiments/runs/federated_align/W.npy"
```
Call `solve_linear_map` on the anchor embeddings, persist `W`, and map incoming queries with `align_vectors` before hitting the remote index.

Unit tests confirm the alignment solver recovers synthetic transforms to within 1e-4 and generalizes to unseen vectors.【F:tests/test_alignment.py†L1-L18】

```python
from embkit.lib.index.alignment import solve_linear_map, align_vectors
W = solve_linear_map(anchor_src, anchor_tgt)
aligned = align_vectors(query_vec[None, :], W)
```

### 14. Similarity Graph + Personalized PageRank (Graph & Structure)

**Use case:** Dense neighbors surface highly similar docs but may miss cluster context; blending a kNN graph with PPR exposes second-order topical connections (e.g., renewable-energy pieces tied to a climate-change article).

**Method:** `build_knn` constructs a symmetric cosine-weighted graph, and `ppr` runs power iteration with restart α to obtain diffusion scores over the graph.【F:embkit/lib/graph/knn.py†L1-L33】【F:embkit/lib/graph/knn.py†L35-L46】 Fuse PPR scores with base similarity to reward nodes close to the query seed and its local cluster.

**Empirical findings:** With α=0.15 we measured cluster recall gains of ~5 points while keeping relevance stable; logs show the chosen α and recall improvements for traceability.【F:experiments/runs/simgraph_ppr/metrics.jsonl†L1-L2】

**Config snippet:**
```yaml
graph:
  enable: true
  k: 10
  alpha: 0.15
```
Enable this block in configs to build the graph alongside the vector search stage.

`tests/test_graph.py::test_ppr_neighbors_rank_higher` ensures the PageRank diffusion ranks local neighbors above unrelated nodes, guarding against normalization regressions.【F:tests/test_graph.py†L1-L12】

### 15. Vector Joins (Structured Multi-query AND)

**Use case:** Compound intents such as "trademark" AND "infringement" need candidates matching all facets; naive averaging favors the dominant term.

**Method:** `vector_join` consumes per-facet result lists (id, score) and boosts items appearing across lists, defaulting to a strict AND before falling back to soft unions; `vector_join_and` enforces intersection-only semantics.【F:embkit/lib/query_ops/__init__.py†L219-L248】 The helper tracks both summed scores and hit counts so you can sort by agreement first, score second.【F:embkit/lib/query_ops/__init__.py†L219-L245】

**Empirical findings:** Joining top-100 lists for legal AND queries improves precision@10 from 0.58→0.70 with 82% of requests yielding a non-empty intersection; metrics are logged for reproducibility.【F:experiments/runs/vector_join/metrics.jsonl†L1-L2】

**Config snippet:**
```python
facets = [search("trademark"), search("infringement")]
joined = vector_join_and(facets)
```

Tests assert that intersections surface first and that the fallback path still returns ranked candidates when no overlap exists.【F:tests/test_vector_join.py†L1-L17】

### 16. Query-Planned Sub-search (Graph & Structure)

**Use case:** Natural-language questions often decompose into entity retrieval plus attribute filtering ("presidents born in New York"). Planning reduces noise by constraining later steps to early results.

**Method:** `planned_search` detects simple "born in" patterns, issues a broad sub-search, and filters the returned documents by location using the helper `extract_born_in`; results fall back gracefully if no matches survive.【F:embkit/lib/planning/__init__.py†L23-L46】

**Empirical findings:** On 50 templated questions we kept precision high by filtering to 4-5 matching entities per query, as recorded in `experiments/runs/query_planning/examples.txt`.【F:experiments/runs/query_planning/examples.txt†L1-L3】

**Config snippet:**
```python
from embkit.lib.planning import planned_search
results = planned_search("presidents born in New York", search_fn)
```

Unit tests cover both the decomposition and filtering logic to guard against regressions.【F:tests/test_planning.py†L1-L18】

### 17. Retrieval Planning for RAG (Generation & Reasoning)

**Use case:** Multi-hop questions need intermediate entities before answering. Planning retrieval steps boosts recall for complex RAG prompts.

**Method:** `multi_hop_retrieve` detects cues like "born in <year>" plus "president of <country>", queries for the entity, extracts a name via `extract_person_name`, then launches a follow-up query scoped to the original question remainder.【F:embkit/lib/planning/__init__.py†L48-L74】

**Empirical findings:** Multi-hop planning lifts single-hop recall from 0.64 to 0.89 on our WikiHop-style sample, as detailed in the planning log.【F:experiments/runs/rag_planning/log.txt†L1-L3】

**Usage snippet:**
```python
from embkit.lib.planning import multi_hop_retrieve
answers = multi_hop_retrieve(question_text, search_fn, k=5)
```

`tests/test_planning.py::test_multi_hop_retrieve_runs_second_query` exercises the two-step planner with synthetic documents to ensure the second query fires and surfaces the final fact.【F:tests/test_planning.py†L20-L33】

### 18. Contextual Compression for Memory (Generation & Reasoning)

**Use case:** RAG pipelines often retrieve more documents than will fit in the generator context; summarizing them into a compact representation preserves coverage while respecting token limits.

**Method:** `average_embedding` produces an L2-normalized mean vector for the top-k retrieved embeddings, while `summarize_sentences` extracts the longest informative sentences to seed textual summaries.【F:embkit/lib/memory/compression.py†L1-L24】 Both feed into downstream prompts or caches and run in milliseconds on CPU.

**Empirical findings:** Averaging the top-5 embeddings keeps cosine ≥0.96 to originals, and compressed-context answers retain 92% recall with a 5x context reduction.【F:experiments/runs/context_compression/log.txt†L1-L5】

**Usage snippet:**
```python
from embkit.lib.memory.compression import average_embedding
comp = average_embedding(doc_vectors[:5])
```

`tests/test_memory.py::test_average_embedding_normalizes_mean` confirms high cosine alignment, and the summarize test ensures the textual helper surfaces informative sentences.【F:tests/test_memory.py†L1-L24】

### 19. kNN Memory Heads (Generation & Reasoning)

**Use case:** Code assistants and QA bots benefit from injecting similar solved examples into the prompt so the generator can mimic known solutions.

**Method:** `build_memory_prompt` appends up to five `[MEM]` hints (pre-summarized) to the user query, reusing `summarize_sentences` when explicit summaries are missing.【F:embkit/lib/memory/knn.py†L1-L16】 You can plug it into any generative model invocation by retrieving nearest neighbors and formatting them as dictionaries.

**Empirical findings:** Memory-augmented runs improve accuracy from 73%→78% overall and nearly guarantee hits when close neighbors exist (97% success).【F:experiments/runs/knn_memory/eval.json†L1-L2】

**Usage snippet:**
```python
prompt = build_memory_prompt(question, neighbors)
response = model.generate(prompt)
```

Tests assert that memory tokens appear in prompts and remain semantically consistent.【F:tests/test_memory.py†L12-L24】

### 20. Counterfactual Query Variations (Generation & Reasoning)

**Use case:** Swapping key query facets (e.g., Germany→Europe) diagnoses which parts of the prompt drive ranking changes and highlights brittle behavior.

**Method:** `generate_counterfactuals` replaces known facets with alternatives, while `rank_delta` measures how ranks shift between result lists.【F:embkit/lib/analysis/counterfactual.py†L1-L23】 Feed the variants through your search pipeline and inspect the deltas to audit sensitivity.

**Empirical findings:** Counterfactual runs report Jaccard overlap and per-document rank shifts; country swaps often halve overlap, signalling location-sensitive behavior.【F:experiments/runs/counterfactual/changes.json†L1-L2】

**Usage snippet:**
```python
from embkit.lib.analysis.counterfactual import generate_counterfactuals
variants = generate_counterfactuals("top universities in Germany", {"Germany": ["Europe", "Asia"]})
```

`tests/test_counterfactual.py` validates facet generation and rank-delta accounting so regressions surface quickly.【F:tests/test_counterfactual.py†L1-L15】

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

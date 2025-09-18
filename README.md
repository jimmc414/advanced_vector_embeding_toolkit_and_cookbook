# An Advanced Vector-Embedding Operations Toolkit and Cookbook

An Advanced Vector-Embedding Operations Toolkit and Cookbook is a CPU-only vector embedding toolkit with FAISS indexing, query ops, graph fusion, safety, calibration, and evaluation.

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

## Hugging Face encoder support (CPU-only)

An Advanced Vector-Embedding Operations Toolkit and Cookbook now ships with a factory that can spin up either the legacy
deterministic `dummy-encoder` or a Hugging Face `sentence-transformers` model
purely on CPU. The CLI keeps backward compatibility—existing configs that only
specify `model.name: dummy-encoder` continue to work and still trigger the
synthetic quickstart path.

### Selecting a model

Choose the provider and friendly name inside your YAML config:

```yaml
model:
  provider: huggingface
  name: e5-mistral         # alias → intfloat/e5-mistral-7b-instruct
  batch_size: 16           # optional, defaults to 32
  max_length: 512          # optional, defaults to 512 tokens
paths:
  corpus: data/my_corpus.jsonl
  embeddings: data/my_embeddings.npy
  output_dir: experiments/runs/my_run
```

The alias registry includes `e5-mistral`, `mxbai-large`, `sfr-mistral`,
`gist-embedding`, and `gte-large`. You can also pass any full
`org/model-name` string directly. When `provider: huggingface` is selected the
index build will batch encode the corpus on CPU, save raw embeddings to the
`paths.embeddings` location, normalize vectors for FAISS, and record metadata in
`build_meta.json` for reproducibility.【F:embkit/cli/index.py†L19-L118】【F:experiments/configs/mteb_e5.yaml†L1-L12】

### Offline caching

Set `model.cache_dir` (forwarded to Hugging Face) plus `paths.embeddings` to
reuse downloads and skip recomputation. If embeddings already exist, the build
command will reuse them; otherwise vectors are computed and persisted for future
runs. When IDs are missing in the corpus we auto-generate deterministic
`doc_00000` style identifiers and write them back to the JSONL so downstream
search has a stable mapping.【F:embkit/cli/index.py†L73-L107】

### CPU resource considerations

Everything runs on CPU. We explicitly cap PyTorch to a single thread and seed it
for deterministic results, so expect encoding throughput comparable to a
single-core CPU inference pass. Install the extra dependencies with `pip install
-r requirements.txt`; the pinned versions correspond to the CPU wheels of
`torch`, `transformers`, `sentence-transformers`, and `huggingface-hub`.【F:embkit/lib/models/hf.py†L26-L86】【F:requirements.txt†L1-L14】

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

### 21. Nullspace Bias Removal (Safety & Fairness)

**Use case:** Audit and mitigate demographic bias in ranked results by removing a protected attribute direction (e.g., gender) from query and document embeddings before scoring.【F:embkit/lib/analysis/nullspace.py†L19-L42】

**Method:** `remove_direction` subtracts the projection of a vector onto the identified bias direction, while `remove_directions` applies the same nullspace projection to batches for efficient re-ranking runs.【F:embkit/lib/analysis/nullspace.py†L10-L42】 Apply the helper to queries and top-k candidates, then compare slates to quantify exposure deltas.

**Empirical findings:** On the jobs audit, projecting out the gender axis doubled the female share in the "programmer" top-5 (20%→50%) with acceptable relevance drift.【F:experiments/runs/nullspace_projection/results.txt†L1-L4】

**Config snippet:**
```python
from embkit.lib.analysis.nullspace import remove_direction, remove_directions
query_db = remove_direction(query_vec, gender_direction)
doc_db = remove_directions(doc_matrix, gender_direction)
```

**Run logs:** Compare before/after rankings under `experiments/runs/nullspace_projection/` to document exposure shifts and decide whether to deploy the debiased embeddings.【F:experiments/runs/nullspace_projection/results.txt†L1-L4】

Unit tests assert that masculine/feminine analogues collapse after projection and that batch and single-vector variants stay aligned.【F:tests/test_nullspace.py†L1-L23】

### 22. Causal Feature Probing with Knockoffs (Generation & Reasoning)

**Use case:** Determine whether a token like "COVID" is causally responsible for high scores or merely correlated noise by generating attribute knockoffs and measuring score deltas.【F:embkit/lib/analysis/knockoff.py†L12-L32】

**Method:** `knockoff_adjust` removes or amplifies an attribute direction inside document vectors, and `knockoff_scores` reports original vs. adjusted cosine scores so you can detect large causal drops.【F:embkit/lib/analysis/knockoff.py†L12-L32】 Remove the attribute for hypothesis tests; add it when crafting counterfactual boosts.

**Empirical findings:** Seven of the top ten medical articles lost ≥10% relevance once "COVID" was knocked out, confirming the retriever's heavy reliance on the term.【F:experiments/runs/causal_probe/report.md†L3-L7】

**Usage snippet:**
```python
from embkit.lib.analysis.knockoff import knockoff_scores
base, adjusted = knockoff_scores(query_vec, doc_vecs, covid_direction)
delta = adjusted - base
```

**Run logs:** The knockoff report records score drops for both true knockoffs and random controls so you can distinguish causal signals from noise.【F:experiments/runs/causal_probe/report.md†L1-L7】

Unit tests cover both vector adjustment and score comparisons to guard against regressions in the causal probe helpers.【F:tests/test_knockoff.py†L1-L18】

### 23. Adversarially Robust Retrieval (Learning & Quality)

**Use case:** Harden retrieval against typos, paraphrases, and gradient-based perturbations so attacks like "capitaloffrance" still return Paris.【F:embkit/lib/training/robust.py†L12-L36】

**Method:** `generate_synonym_variants` creates paraphrased queries for augmentation, while `fgsm_perturb` applies normalized FGSM noise during training to simulate worst-case embedding shifts.【F:embkit/lib/training/robust.py†L12-L36】 Integrate both into your training loop and certify that perturbed queries keep retrieving the correct document.

**Empirical findings:** The robust model lifted Recall@1 from ~0.5→0.88 on typo and synonym attacks and held up under ε=0.1 FGSM probes.【F:experiments/runs/robust_eval/summary.txt†L1-L5】

**Config snippet:**
```python
variants = generate_synonym_variants(query_text, synonym_map)
adv_query = fgsm_perturb(query_vec, grad, epsilon=0.1)
```

**Run logs:** Attack sweeps in `experiments/runs/robust_eval/summary.txt` summarize baseline vs. robust recall so you can quantify resilience gains.【F:experiments/runs/robust_eval/summary.txt†L1-L5】

Unit tests verify paraphrase generation, FGSM normalization, and the surrounding training utilities so robustness tooling stays reliable.【F:tests/test_training_ops.py†L1-L33】

### 24. Hard Negative Mining (Learning & Quality)

**Use case:** Improve precision by mining high-scoring false positives and enforcing margin losses that push them beneath the correct answer.【F:embkit/lib/training/hard_negative.py†L10-L31】

**Method:** Use `mine_hard_negatives` to collect the strongest mistakes from ranked slates, then feed those triplets through `triplet_margin` (or your optimizer of choice) during re-training.【F:embkit/lib/training/hard_negative.py†L10-L31】 Refresh the mined pool every epoch or two for continuous improvement.

**Empirical findings:** Two mining rounds demoted 85% of the hardest negatives and raised FAQ nDCG@10 to 0.912.【F:experiments/runs/training/hard_neg_mining.log†L1-L3】【F:experiments/runs/training/metrics_final.json†L1-L1】

**Usage snippet:**
```python
hard = mine_hard_negatives(ranked_ids, positives={pos_id}, limit=3)
loss = triplet_margin(q_vec, pos_vec, neg_vec)
```

**Run logs:** Mining summaries and final metrics under `experiments/runs/training/` document coverage and quality uplift for audits.【F:experiments/runs/training/hard_neg_mining.log†L1-L3】【F:experiments/runs/training/metrics_final.json†L1-L1】

Unit tests ensure positives are never mistaken for negatives and that triplet margins drop to zero once the positive outranks the negative.【F:tests/test_training_ops.py†L23-L33】

### 25. Active Feedback Labeling (Learning & Quality)

**Use case:** Focus human labeling budget on uncertain queries by measuring ranking margins and triaging low-confidence searches.【F:embkit/lib/training/active.py†L10-L27】

**Method:** `margin_uncertainty` computes the top-1 vs. top-2 gap, and `select_uncertain_queries` returns query IDs below a configurable threshold for annotation.【F:embkit/lib/training/active.py†L10-L27】 Feed the resulting list to your labeling UI or feedback queue.

**Empirical findings:** Day-one triage flagged queries with <0.05 margin and, after retraining, boosted MRR on that cohort from 0.45→0.60.【F:experiments/runs/active_learning/day1.log†L1-L4】【F:experiments/runs/active_learning/metrics_active.json†L1-L1】

**Usage snippet:**
```python
targets = select_uncertain_queries(score_map, threshold=0.1, limit=100)
```

**Run logs:** Daily selection logs plus `metrics_active.json` capture which queries were labeled and the resulting gains for reporting to stakeholders.【F:experiments/runs/active_learning/day1.log†L1-L4】【F:experiments/runs/active_learning/metrics_active.json†L1-L1】

Unit tests assert that margin math behaves as expected and that the selector surfaces only low-confidence queries.【F:tests/test_training_ops.py†L36-L39】

### 26. Drift Detection & Model Update (Learning & Quality)

**Use case:** Detect corpus/query distribution shifts (e.g., "metaverse" spikes) early and trigger index/model refreshes before quality drops.【F:embkit/lib/monitoring/drift.py†L8-L30】

**Method:** `hotelling_t2` compares the current batch mean against historical statistics with a regularized Hotelling's T², and `detect_drift` flags alarms when scores exceed your threshold.【F:embkit/lib/monitoring/drift.py†L8-L30】 Combine with monitoring to decide when to re-train or ingest new content.

**Empirical findings:** A May 1st surge produced T²=15.2 (>10 threshold) and triggered an update; metrics normalized after refresh.【F:experiments/runs/drift_detection/log.txt†L1-L2】【F:experiments/runs/drift_detection/metrics_may_2025.json†L1-L1】

**Usage snippet:**
```python
alert = detect_drift(new_query_batch, ref_mean, ref_cov, threshold=10.0)
```

**Run logs:** Drift logs and post-update metrics provide traceability for every alert and remediation action.【F:experiments/runs/drift_detection/log.txt†L1-L2】【F:experiments/runs/drift_detection/metrics_may_2025.json†L1-L1】

Unit tests cover both high-drift and steady-state scenarios so alarms remain trustworthy.【F:tests/test_monitoring.py†L1-L20】

### 27. Score Calibration and Confidence (Learning & Quality)

**Use case:** Produce calibrated relevance probabilities for downstream consumers by correcting overconfident cosine scores.【F:embkit/lib/calibrate/temperature.py†L6-L45】【F:embkit/lib/calibrate/isotonic.py†L8-L44】

**Method:** Fit a temperature parameter with `temperature_fit` for quick global scaling, or extract a monotonic mapping via `isotonic_fit`/`isotonic_apply` when you need non-linear calibration.【F:embkit/lib/calibrate/temperature.py†L6-L45】【F:embkit/lib/calibrate/isotonic.py†L8-L44】 Apply the calibrated probabilities to your API responses.

**Empirical findings:** Temperature scaling (T≈1.5) cut ECE to 0.02 and isotonic smoothing squeezed it a bit further without harming ranking quality.【F:experiments/runs/calibration/ece_brier.txt†L1-L3】

**Usage snippet:**
```python
T = temperature_fit(labels, logits)
p_cal = temperature_apply(logits, T)
thr, val = isotonic_fit(labels, logits)
p_iso = isotonic_apply(logits, thr, val)
```

**Run logs:** Calibration dashboards live under `experiments/runs/calibration/`, recording ECE/Brier improvements per method for audit trails.【F:experiments/runs/calibration/ece_brier.txt†L1-L3】

Unit tests confirm both temperature and isotonic calibrators behave and remain monotonic.【F:tests/test_calibration.py†L1-L36】

### 28. Fairness-Aware Re-ranking (Safety & Fairness)

**Use case:** Balance exposure between protected and majority groups when presenting ranked results (e.g., author demographics in publication search).【F:embkit/lib/query_ops/__init__.py†L201-L247】

**Method:** `fair_rerank` greedily assembles a slate that meets the desired protected-group ratio while honoring relevance when both groups have supply.【F:embkit/lib/query_ops/__init__.py†L201-L247】 Tune `target_ratio` to match candidate share and set `top_k` to your display length.

**Empirical findings:** Re-ranking moved minority-serving institutions from 12%→31% exposure in the top-10 with only a 1.5 point nDCG drop.【F:experiments/runs/fairness/rerank_log.txt†L1-L4】 The aggregate metrics show exposure aligning to 29/71 with strong overall relevance.【F:experiments/runs/fairness/metrics_fairness.json†L1-L1】

**Usage snippet:**
```python
slate = fair_rerank(candidate_ids, relevance_scores, group_labels, protected_group="A", target_ratio=0.3, top_k=10)
```

**Run logs:** Fairness dashboards log per-query exposure deltas so you can monitor the relevance–fairness trade-off across releases.【F:experiments/runs/fairness/rerank_log.txt†L1-L4】【F:experiments/runs/fairness/metrics_fairness.json†L1-L1】

Unit tests guarantee the helper meets minimum protected coverage targets in prefix positions.【F:tests/test_fairness.py†L1-L12】

### 29. Privacy-Preserving Retrieval (Safety & Privacy)

**Use case:** Protect sensitive queries (e.g., medical lookups) by transforming embeddings client-side and training with differential privacy noise.【F:embkit/lib/safety/privacy.py†L10-L35】

**Method:** `apply_secure_transform` projects local embeddings through a learned matrix and optional noise before sending them to the server, while `dp_gaussian_noise` clips and perturbs gradients during DP-SGD fine-tuning.【F:embkit/lib/safety/privacy.py†L10-L35】 Together they limit information leakage in transit and training.

**Empirical findings:** The privacy audit logged ε=8, δ=1e-6 with membership-inference AUC ≈0.52 and only ~2% recall loss relative to non-DP training.【F:experiments/runs/privacy/privacy_audit.json†L1-L1】

**Usage snippet:**
```python
secure = apply_secure_transform(local_embed, transform_matrix, noise_scale=1e-3)
dp_grad = dp_gaussian_noise(gradients, clip_norm=1.0, noise_multiplier=0.8)
```

**Run logs:** Privacy reports document DP budgets, audit scores, and quality deltas for compliance reviews.【F:experiments/runs/privacy/privacy_audit.json†L1-L1】

Unit tests cover normalization of the secure transform and DP clipping to prevent regressions that could weaken guarantees.【F:tests/test_privacy.py†L1-L18】

### 30. PII Screening and Redaction (Safety & Privacy)

**Use case:** Detect and redact structured PII (emails, SSNs, phone numbers) before exposing snippets to end users.【F:embkit/lib/safety/pii.py†L6-L35】

**Method:** `pii_contains` and `pii_redact` rely on curated regexes, while `pii_filter_results` walks result dictionaries to mask any offending fields by default.【F:embkit/lib/safety/pii.py†L6-L35】 Adjust the field or token to match your UI requirements.

**Empirical findings:** Screening cut exposed PII by 68% while keeping nDCG loss to ~1%.【F:experiments/runs/pii_filter/log.txt†L1-L2】【F:experiments/runs/pii_filter/metrics_pii.json†L1-L1】

**Usage snippet:**
```python
safe_results = pii_filter_results(search_results)
```

**Run logs:** PII filter logs list every redaction and its reason so policy teams can audit changes over time.【F:experiments/runs/pii_filter/log.txt†L1-L2】

Unit tests ensure regex coverage and result masking remain intact as policies evolve.【F:tests/test_safety.py†L1-L25】


### 31. Counterfactual Sensitivity Probing (Analysis & Quality)

**Use case:** When auditing ranking changes you often want to know how swapping a geographic facet or time window affects the slate. Counterfactual probing automatically rewrites the query and measures the rank deltas so you can flag regressions before launch.【F:embkit/lib/analysis/counterfactual.py†L1-L25】

**Method:** `generate_counterfactuals` replaces facet tokens with alternatives (e.g., "Germany"→"Europe"), then `rank_delta` compares the original and counterfactual rankings to highlight documents that moved sharply.【F:embkit/lib/analysis/counterfactual.py†L1-L25】 The helpers slot directly into the evaluation harness so probes live alongside your usual metrics.

**Empirical findings:** On our geography audit, counterfactual rewrites surfaced moderate rank churn (Jaccard overlap 0.18–0.42) that product owners reviewed before promotion.【F:experiments/runs/counterfactual/changes.json†L1-L2】

**Workflow snippet:**
```python
from embkit.lib.analysis.counterfactual import generate_counterfactuals, rank_delta
cf = generate_counterfactuals("top universities in Germany", {"Germany": ["Europe", "Asia"]})
diffs = rank_delta(base_slate, rerun_search(cf[0]))
```

Unit tests keep both the query generator and delta calculator honest so regression suites fail if counterfactual coverage drops.【F:tests/test_counterfactual.py†L1-L17】

### 32. Polytope Queries (Advanced Query Logic)

**Use case:** Recruiters and marketplace operators need an "AND" over multiple learned attributes—e.g., >5 years experience **and** Python **and** NYC—without relying on brittle metadata flags.

**Method:** `polytope_filter` applies a list of half-space constraints over normalized document vectors, returning only items that satisfy every threshold.【F:embkit/lib/query_ops/__init__.py†L116-L126】 Pair it with facet-specific centroids to express experience, skill, and location as learned directions.【F:tests/test_query_ops.py†L44-L49】

**Empirical findings:** Tightening the years/skill/location thresholds shrank the candidate pool from 48 to 8 while preserving the target talent list in our hiring benchmark.【F:experiments/runs/polytope_query/metrics.jsonl†L1-L2】

**Config snippet:**
```python
constraints = [
    (years_exp_vec, 0.5),
    (python_skill_vec, 0.6),
    (nyc_location_vec, 0.55),
]
keep = polytope_filter(doc_matrix, constraints)
```

### 33. Cone and Sector Queries (Advanced Query Logic)

**Use case:** Scholars asking for "papers more about quantum computing than classical" want to bound results to a semantic cone around a preferred topic while excluding competing directions.

**Method:** `cone_filter` screens candidates by cosine angle, effectively enforcing an angular sector around the query direction; combine it with subtractive negatives to exclude opposing topics.【F:embkit/lib/query_ops/__init__.py†L116-L118】 The regression test demonstrates that off-axis documents are removed once the cosine floor is raised.【F:tests/test_query_ops.py†L32-L40】

**Empirical findings:** Raising the cosine floor from 0.70 to 0.85 cut the candidate set in half while bumping the "quantum-first" hit rate from 0.78 to 0.91 on our topical benchmark.【F:experiments/runs/cone_sector/metrics.jsonl†L1-L2】

**Code snippet:**
```python
hits = cone_filter(query_vec, doc_matrix, cos_min=0.85)
```

### 34. Learned Metric Retrieval (Mahalanobis Distance)

**Use case:** Visual search often needs to overweight color or other salient cues so near-duplicate items outrank merely related ones.

**Method:** `mahalanobis_diag` evaluates the learned diagonal metric (feature weights) during re-ranking, letting you plug weights straight from a metric-learning job into production scoring.【F:embkit/lib/query_ops/__init__.py†L128-L131】 The unit test guards both the positive-weight constraint and the relative scaling behavior.【F:tests/test_query_ops.py†L140-L150】

**Empirical findings:** A diagonal weighting that tripled the color dimension lifted top-5 precision from 0.72 to 0.79 on our product catalog while maintaining stability for other features.【F:experiments/runs/metric_learning/metrics.jsonl†L1-L2】

**Code snippet:**
```python
d = mahalanobis_diag(query_vec, candidate_vec, learned_weights)
```

### 35. Hyperbolic Embeddings for Hierarchy (Representation & Indexing)

**Use case:** Taxonomy-heavy corpora (Wikipedia categories, org charts) benefit from hyperbolic geometry, which expands space near the boundary to represent broad-vs-specific relationships.

**Method:** `poincare_distance` computes the Poincaré-ball geodesic after optionally projecting vectors back inside the unit ball, giving you a drop-in distance function for hierarchical retrieval.【F:embkit/lib/utils/hyperbolic.py†L1-L25】 Tests confirm distances stretch more aggressively near the boundary, matching hyperbolic intuition.【F:tests/test_hyperbolic.py†L1-L19】

**Empirical findings:** Switching the distance metric lifted Mean Average Precision on high-level category queries from 0.67 to 0.82 because descendants stayed tightly clustered around their parents.【F:experiments/runs/hyperbolic_eval/metrics.jsonl†L1-L2】

**Code snippet:**
```python
dist = poincare_distance(query_vec, doc_vec)
```

### 36. Disentangled Subspace Representations (Learning & Quality)

**Use case:** Creative tools and recommendation experiences often need to swap "style" while holding "content" fixed (or vice versa) for interactive filtering.

**Method:** `split_embedding`, `swap_subspace`, and `merge_embedding` slice a vector into named factors and recombine them so you can transplant styles or contents between exemplars before searching.【F:embkit/lib/analysis/disentangle.py†L1-L42】 Tests cover both round-trip reconstruction and safe swapping to guard against index errors.【F:tests/test_disentangle.py†L1-L21】

**Empirical findings:** Keeping style rank at 2–4 dimensions preserved ≥0.90 precision for both style and content facets on our multimedia set, enabling controllable retrieval without bloating vector size.【F:experiments/runs/disentangled_subspace/metrics.jsonl†L1-L2】

**Code snippet:**
```python
style_q, content_q = split_embedding(query_vec, [128, 128])
remixed = merge_embedding([style_reference, content_q])
```

### 37. Mixture-of-Experts Encoders (Representation & Indexing)

**Use case:** Mixed-domain search (code + natural language) benefits from routing each query to the right expert encoder so domain-specific nuance is preserved.

**Method:** `KeywordExpertRouter` scores queries against per-expert keyword lists while `combine_expert_embeddings` blends outputs when soft routing is desired.【F:embkit/lib/models/mixture.py†L1-L52】 Unit tests validate both the routing preference and the weighted combination math.【F:tests/test_mixture.py†L1-L21】

**Empirical findings:** The mixture routed code queries to the specialized encoder 95% of the time, boosting code MRR from 0.50 to 0.65 while inching text MRR from 0.58 to 0.61 thanks to gentler blending.【F:experiments/runs/mixture_experts/metrics.jsonl†L1-L2】

**Code snippet:**
```python
router = KeywordExpertRouter({"code": ["def", "class"], "text": ["news", "article"]}, default_expert="text")
expert, _ = router.route(query)[0]
emb = encoders[expert].encode(query)
```

### 38. Streaming Index with TTL (Ops & Freshness)

**Use case:** News and alerting systems require that fresh stories become searchable immediately while stale content quietly ages out.

**Method:** `StreamingIndex` keeps a normalized in-memory list of `(vector, id, timestamp)` records, serves cosine searches, and exposes `prune_expired` for TTL-based eviction.【F:embkit/lib/index/streaming.py†L1-L48】 Tests exercise both retrieval quality and the eviction path to ensure expired items disappear deterministically.【F:tests/test_streaming_index.py†L1-L28】

**Empirical findings:** With a 7-day active window the streaming index hit 0.92 freshness-weighted Recall@20 while keeping p95 latency at 42 ms; widening to 30 days broadened recall slightly but still cleared latency budgets.【F:experiments/runs/streaming_index/metrics.jsonl†L1-L2】

**Code snippet:**
```python
idx = StreamingIndex(dim=doc_matrix.shape[1])
idx.add(batch_vectors, batch_ids, batch_timestamps)
idx.prune_expired(time.time(), ttl_seconds=30*86400)
```

### 39. Generative Latent Retrieval Hybrid (RAG & IR)

**Use case:** Combining generative retrievers (DSI/seq2seq) with dense re-rankers recovers long-tail matches while keeping precision high.

**Method:** `merge_generative_dense` normalizes generative ID scores and dense similarities before blending them with a tunable weight, returning a fused ranked list.【F:embkit/lib/query_ops/__init__.py†L254-L294】 Tests cover both the shared-hit boost and empty-input edge cases so integration remains stable.【F:tests/test_generative.py†L1-L14】

**Empirical findings:** Feeding 100 generative IDs into the hybrid raised Recall@100 from 0.83 to 0.85 with only ~21 ms additional rerank latency once the dense pass rescored the union.【F:experiments/runs/generative_hybrid/metrics.txt†L1-L4】

**Code snippet:**
```python
fused = merge_generative_dense(generative_candidates, dense_scores, weight=0.6, top_k=50)
```

### 40. Cross-lingual Alignment with Regularization (Learning & Quality)

**Use case:** Multilingual users expect Spanish queries to surface English documents seamlessly.

**Method:** We learn a linear alignment matrix with optional regularization using `solve_linear_map`/`align_vectors`, then monitor Frobenius alignment error to ensure English performance stays intact.【F:embkit/lib/index/alignment.py†L1-L23】 The alignment regression test recreates a synthetic transform to verify the solver and mapper stay precise.【F:tests/test_alignment.py†L1-L17】

**Empirical findings:** Light regularization and pseudo-bitext lifted Spanish→English Recall@5 from 0.50 to 0.80 while holding monolingual accuracy within 1%.【F:experiments/runs/cross_lingual_alignment/metrics.jsonl†L1-L2】【F:experiments/runs/cross_lingual_alignment/log.txt†L1-L1】

**Code snippet:**
```python
W = solve_linear_map(src_matrix, tgt_matrix)
aligned = align_vectors(new_queries, W)
```

### 41. Knowledge-Graph Fused Retrieval (Multimodal & Structured)

**Use case:** Biomedical and enterprise search benefit from traversing known relationships (gene→drug→paper) in parallel with vector similarity so synonyms and related concepts surface.

**Method:** `meta_path_scores` walks typed edges along a configurable meta-path while `fuse_with_graph` adds the KG-derived boost back into the dense score table.【F:embkit/lib/graph/kg.py†L1-L48】 Tests ensure connected meta-paths receive positive weight and boosted documents rise to the top.【F:tests/test_kg.py†L1-L21】

**Empirical findings:** The KG trace for "EGFR cancer therapy" highlighted pathway-targeted trials that pure vector search missed, enriching the top results with domain-critical context.【F:experiments/runs/kg_fusion/examples.md†L1-L6】

**Code snippet:**
```python
kg_scores = meta_path_scores(kg_graph, seeds=["gene:egfr"], meta_path=["gene", "drug", "paper"], decay=0.5)
reranked = fuse_with_graph(base_scores, kg_scores, weight=0.3)
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

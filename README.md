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

All ten scenarios from the Vector Embedding Innovations write-up are wired into `embkit.lib.query_ops`. Each recipe ships with a runnable snippet, configuration pointers, and a regression test so you can reproduce the behaviour end-to-end once `make setup` has installed the dependencies.

### 1. Directional Semantic Search (Query Logic)

**Use case.** Steer results toward a user preference such as _“more about policy than politics”_ when ranking news articles. The direction vector represents the desired trait (policy) so the final query is nudged in that semantic direction.【F:embkit/lib/query_ops/__init__.py†L11-L24】

**Method.** The helper normalizes `q + α·v_dir` before computing cosine similarity against document vectors. Sweeping `α` in {0.2, 0.5, 0.8, 1.0} typically shows that moderate weights shift rankings without drowning out the base intent; the unit test confirms that increasing `α` elevates the direction-aligned document.【F:tests/test_query_ops.py†L22-L33】

**Try it yourself.**

```python
import numpy as np
from embkit.lib.query_ops import directional_search

q = np.array([1.0, 0.1], dtype=np.float32)
direction = np.array([0.0, 1.0], dtype=np.float32)
docs = np.array([[0.9, 0.0], [0.5, 0.5]], dtype=np.float32)

baseline = directional_search(q, np.zeros_like(direction), docs, alpha=0.0)
biased = directional_search(q, direction, docs, alpha=0.8)
print(baseline[0], biased[0])  # -> 0 then 1 once the policy direction is applied
```

**Config hooks.** Add a block like the following to any YAML config under `query_ops` to bias the demo search step; the default `experiments/configs/demo.yaml` already enables it.

```yaml
query_ops:
  - op: "directional"
    alpha: 0.8
    vector_path: "experiments/vectors/policy_direction.npy"
```

If `vector_path` is omitted the CLI will fall back to an all-ones vector for experimentation.【F:embkit/cli/search.py†L52-L74】【F:experiments/configs/demo.yaml†L10-L16】

**Validation.** `tests/test_query_ops.py::test_directional_shifts_rank` locks in the expected reordering so changes to the scoring logic cannot silently regress.【F:tests/test_query_ops.py†L22-L33】

### 2. Contrastive Query Filtering (Query Logic)

**Use case.** Filter out e-commerce listings that match an undesired concept such as _“affordable laptop, not used”_ by demoting results aligned with a “used goods” vector.【F:embkit/lib/query_ops/__init__.py†L23-L35】

**Method.** The contrastive scorer subtracts `λ·cos(doc, v_neg)` from the base similarity, effectively carving out a semantic cone that excludes unwanted traits. Dial up `λ` toward 1.0 for stricter filtering; the regression test verifies that a “used” document receives a lower score than the pristine counterpart.【F:tests/test_query_ops.py†L45-L55】

**Try it yourself.**

```python
import numpy as np
from embkit.lib.query_ops import contrastive_score

q = np.array([0.7, 0.7], dtype=np.float32)
neg = np.array([0.0, 1.0], dtype=np.float32)
doc_ok = np.array([0.6, 0.0], dtype=np.float32)
doc_bad = np.array([0.1, 0.9], dtype=np.float32)
print(contrastive_score(q, neg, doc_ok, lam=0.5) > contrastive_score(q, neg, doc_bad, lam=0.5))
```

**Config hooks.** Insert an operation into your `query_ops` chain to penalize a stored negative vector:

```yaml
  - op: "contrastive"
    lambda: 0.5
    vector_path: "experiments/vectors/used_goods.npy"
```

The CLI loads the vector from disk if present and falls back to ones for quick demos, letting you prototype without extra assets.【F:embkit/cli/search.py†L83-L92】

**Validation.** See `tests/test_query_ops.py::test_contrastive_penalizes_negative_concept` for the guardrail that keeps the exclusion weight behaving as expected.【F:tests/test_query_ops.py†L45-L55】

### 3. Compositional (Multi-Facet) Search (Query Logic)

**Use case.** Retrieve academic articles or reports that must mention multiple facets—e.g., _“machine learning AND healthcare”_—without relying on keyword intersections.【F:embkit/lib/query_ops/__init__.py†L51-L67】

**Method.** `facet_subsearch` embeds each facet separately, pulls top-k candidates per facet, and re-ranks by aggregated cosine scores so documents covering all facets bubble up. Auxiliary helpers such as `compose_and` and `polytope_filter` let you merge aligned vectors or enforce additional cone constraints for stricter logical AND behaviour.【F:embkit/lib/query_ops/__init__.py†L37-L67】【F:embkit/lib/query_ops/__init__.py†L147-L155】

**Try it yourself.**

```python
import numpy as np
from embkit.lib.query_ops import facet_subsearch

docs = np.array([[0.9, 0.8], [0.95, 0.1], [0.1, 0.95], [0.5, 0.5]], dtype=np.float32)
ml = np.array([1.0, 0.0], dtype=np.float32)
health = np.array([0.0, 1.0], dtype=np.float32)
ranked = facet_subsearch([ml, health], docs, top_k=3)
print(ranked)  # multi-facet documents rank highest
```

**Config hooks.** For production you typically run separate ANN lookups per facet and merge the IDs before handing them to `facet_subsearch`. Within the demo pipeline you can load precomputed facet vectors and perform the merge during post-processing.

**Validation.** `test_facet_subsearch_prioritizes_multi_facet_docs` confirms that documents covering both sub-queries outrank single-facet items in the reference scenario.【F:tests/test_query_ops.py†L57-L68】

### 4. Analogical Search (Query Logic)

**Use case.** Solve analogy-style questions such as _“France is to Paris as Japan is to ?”_ or _“framework : maintainer :: library : ?”_ in a knowledge base.【F:embkit/lib/query_ops/__init__.py†L69-L75】

**Method.** `analogical_query` performs the classic `b - a + c` vector arithmetic while keeping the result normalized. Plug it into your ANN index to retrieve the nearest neighbour, optionally excluding the prompt tokens to avoid trivial matches.【F:embkit/lib/query_ops/__init__.py†L69-L75】

**Try it yourself.**

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

After computing `vec`, search for the closest embedding to retrieve the answer.

**Config hooks.** Analogy mode is typically run ad-hoc: fetch the base pair `(A, B)` from metadata, compute `analogical_query(A, B, query_term)`, and pass the resulting vector into your standard retrieval call.

**Validation.** The `test_analogical_query_returns_expected_vector` regression uses the classic king/queen example to ensure the helper maintains the expected directionality.【F:tests/test_query_ops.py†L70-L82】

### 5. Diversity-Aware Re-ranking (MMR/DPP) (Query Logic)

**Use case.** Prevent the top results for _“JavaScript frameworks”_ from being dominated by a single framework by rewarding novelty during re-ranking.【F:embkit/lib/query_ops/__init__.py†L77-L99】

**Method.** `mmr_select` iteratively picks the document that maximizes `λ·sim(doc, query) - (1-λ)·max_{sel} sim(doc, selected)` so each additional item balances relevance and diversity. Lower `λ` values emphasise diversity; higher values stay close to the baseline ranking.【F:embkit/lib/query_ops/__init__.py†L77-L99】

**Try it yourself.**

```python
import numpy as np
from embkit.lib.query_ops import mmr_select

q = np.random.rand(5).astype(np.float32)
docs = np.stack([q + 0.1, q + 0.1, q * 0.8]).astype(np.float32)
chosen = mmr_select(q, docs, k=2, lam=0.5)
print(chosen)  # returns two distinct indices
```

**Config hooks.** Append an MMR step after retrieval so only the top-k candidates are re-ordered. The demo configuration reranks the post-directional candidates before writing search results.【F:experiments/configs/demo.yaml†L14-L16】【F:embkit/cli/search.py†L75-L82】

```yaml
  - op: "mmr"
    lambda: 0.7
    k: 10
```

**Validation.** `test_mmr_select_promotes_diversity` asserts that the second pick differs from the first, protecting against regressions that would surface duplicate content.【F:tests/test_query_ops.py†L84-L98】

### 6. Temporal Decay Ranking (Query Logic)

**Use case.** Highlight fresher stories for recency-sensitive topics such as _“COVID travel restrictions”_ without completely burying authoritative evergreen pieces.【F:embkit/lib/query_ops/__init__.py†L100-L106】

**Method.** Apply an exponential decay `score·exp(-γ·age_days)` either per-document (`temporal_score`) or across a candidate vector (`temporal`). Choose `γ` to match the desired half-life; for example, `γ=0.01` corresponds to roughly a 70-day half-life.【F:embkit/lib/query_ops/__init__.py†L100-L106】

**Try it yourself.**

```python
import numpy as np
from embkit.lib.query_ops import temporal, temporal_score

scores = np.array([0.8, 0.8], dtype=np.float32)
ages = np.array([10.0, 100.0], dtype=np.float32)
print(temporal(scores, ages, gamma=0.02))
print(temporal_score(0.8, 10.0, gamma=0.02))
```

Expect the younger document (10 days old) to retain a higher adjusted score.

**Config hooks.** After retrieving the top candidates you can call `temporal` before the final write step—e.g., by wrapping it inside a custom pipeline stage that multiplies your ANN scores by the decay factor derived from timestamps.

**Validation.** The regression test `test_temporal_decay_prefers_recent_docs` demonstrates that recent items outrank stale ones at identical base scores.【F:tests/test_query_ops.py†L100-L111】

### 7. Personalized Search (Query Logic)

**Use case.** Rerank shared queries such as _“jazz concerts”_ differently per user by incorporating a profile embedding (for example, a user who frequently attends sports events vs. theatre).【F:embkit/lib/query_ops/__init__.py†L108-L114】

**Method.** `personalize` linearly combines query-doc and user-doc cosine similarities via a tunable `β`. You can precompute user vectors offline and load them alongside the request; fall back to `β=0` for cold-start users.【F:embkit/lib/query_ops/__init__.py†L108-L114】

**Try it yourself.**

```python
import numpy as np
from embkit.lib.query_ops import personalize, personalized_score

q = np.array([0.0, 1.0], dtype=np.float32)
user = np.array([1.0, 0.0], dtype=np.float32)
docs = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
print(personalize(q, user, docs, beta=0.5))
print(personalized_score(q, user, docs[0], beta=0.5))
```

**Config hooks.** Personalization is usually implemented at the API layer: pull the user embedding from storage, call `personalize` on the candidate matrix, then trim to the desired depth.

**Validation.** `test_personalization_boosts_profile_matches` asserts that documents aligned with the user profile receive higher scores, preventing regressions that might break targeting.【F:tests/test_query_ops.py†L113-L124】

### 8. Multi-Vector Late Interaction (Representation & Indexing)

**Use case.** Increase recall for Q&A style queries such as _“What is the capital of Australia?”_ by allowing different query tokens to match different document passages.【F:embkit/lib/query_ops/__init__.py†L134-L144】

**Method.** `late_interaction_score` mirrors the ColBERT MaxSim operator: normalize every query and document vector, compute the maximum similarity per query token, then sum the maxima to produce the final score. This preserves fine-grained matching while remaining compatible with ANN indexes that store multiple vectors per document.【F:embkit/lib/query_ops/__init__.py†L134-L144】

**Try it yourself.**

```python
import numpy as np
from embkit.lib.query_ops import late_interaction_score

q_tokens = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
doc_tokens = np.array([[0.9, 0.1], [0.1, 0.95], [0.4, 0.4]], dtype=np.float32)
print(late_interaction_score(q_tokens, doc_tokens))
```

**Config hooks.** Supply query-token and document-token embeddings from your encoder (e.g., ColBERT or a custom late-interaction model) and call the helper during reranking. Cap the number of vectors per document to balance accuracy and latency.

**Validation.** `test_late_interaction_score_rewards_term_coverage` locks in the expectation that documents covering all query facets outscore ones that only match a subset.【F:tests/test_query_ops.py†L126-L135】

### 9. Attribute Subspace Retrieval (Representation & Indexing)

**Use case.** Enforce strict facet compliance—e.g., ensuring that _“4K OLED TV”_ results all share the requested resolution and panel type—by filtering on disentangled attribute dimensions.【F:embkit/lib/query_ops/__init__.py†L147-L155】

**Method.** `subspace_similarity` masks vectors to a chosen attribute subspace before computing cosine similarity. Combine multiple attribute constraints with `polytope_filter` to form an intersection of half-spaces, approximating an AND over attributes.【F:embkit/lib/query_ops/__init__.py†L147-L155】【F:embkit/lib/query_ops/__init__.py†L120-L126】

**Try it yourself.**

```python
import numpy as np
from embkit.lib.query_ops import subspace_similarity

mask = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # focus on resolution dimension
q = np.array([0.9, 0.2, 0.1], dtype=np.float32)
doc = np.array([0.8, -0.1, 0.0], dtype=np.float32)
print(subspace_similarity(q, doc, mask))
```

For stricter gating, build a list of `(vector, threshold)` pairs and pass them to `polytope_filter` before scoring.

**Config hooks.** Persist learned attribute masks (for example, via SVD or supervised factorisation) alongside the index and apply them as a pre-filter before scoring. The helper operates purely on NumPy arrays, so it can be slotted into any pipeline stage.

**Validation.** `test_subspace_similarity_filters_by_attribute` and `test_polytope_intersection` guarantee that only documents satisfying every attribute constraint survive filtering.【F:tests/test_query_ops.py†L137-L155】

### 10. Hybrid Sparse + Dense Retrieval (Representation & Indexing)

**Use case.** Merge lexical BM25 and dense semantic scores to capture both rare keyword matches and semantic paraphrases when doing web or enterprise search.【F:embkit/lib/query_ops/__init__.py†L158-L170】

**Method.** `hybrid_score_mix` normalizes each score list to [0,1] and linearly interpolates them using weight `w`. Set `w≈0.6` to favour sparse signals on encyclopedia-style corpora, or tilt toward dense scores for conversational datasets.【F:embkit/lib/query_ops/__init__.py†L158-L170】

**Try it yourself.**

```python
from embkit.lib.query_ops import hybrid_score_mix

bm25 = [2.0, 1.5, 0.5]
dense = [0.2, 0.7, 0.9]
print(hybrid_score_mix(bm25, dense, weight=0.6))
```

**Config hooks.** Align document IDs from both retrieval stages, then fuse the normalized scores before producing the final ranking. The helper returns a list so you can feed the fused scores back into your ranking logic of choice.

**Validation.** `test_hybrid_score_mix_handles_extreme_weights` checks both sparse-only (`w=0`) and dense-only (`w=1`) extremes to ensure the interpolation behaves predictably.【F:tests/test_query_ops.py†L157-L169】
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

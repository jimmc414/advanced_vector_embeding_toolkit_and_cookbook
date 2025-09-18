# embkit Requirements

## 1. Purpose and Scope

* The system **shall** implement a modular vector-embedding toolkit that goes beyond plain similarity search.
* The toolkit **shall** include core models, FAISS indexing, query algebra operations, graph fusion, safety, calibration, evaluation, CLIs, tests, and a tiny quickstart dataset.
* The v1 scope **shall** target single-host CPU, Python 3.11, float32 everywhere, and no internet calls.

## 2. Definitions

* **Docs matrix**: `D ∈ ℝ^{n×d}` float32, L2-normalized row-wise.
* **Query vector**: `q ∈ ℝ^{d}` float32, L2-normalized.
* **Multi-vector doc**: `list[np.ndarray (m_i,d)]`, each row L2-normalized.
* **Result**: `{"id": str, "score": float}`.

## 3. Preconditions

* The runtime environment **shall** be Python 3.11.
* The project **shall** pin and install: `numpy`, `scipy`, `scikit-learn`, `faiss-cpu`, `networkx`, `pydantic`, `pyyaml`, `typer`, `pytest`, `tqdm`.
* The system **shall not** perform any external network calls.
* If no corpus exists on disk, the system **shall** generate a synthetic demo (200 docs, 2D features, timestamps) under `data/tiny/`.

## 4. Determinism and Reproducibility

* The system **shall** set all seeds deterministically (Python `random`, NumPy, scikit-learn).
* KMeans and related scikit-learn routines **shall** fix `n_init` explicitly.
* The system **shall** set single-threaded BLAS where possible (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) at process start.
* All randomized steps (sampling, shuffling, synthetic generation) **shall** be seeded and logged.
* All persisted artifacts **shall** include a metadata file with seed, package versions, and config hash.

## 5. Constraints

* All embeddings used for cosine/IP search **must** be L2-normalized.
* All arrays **must** be float32.
* Public APIs **shall** be pure functions where practical and **shall not** rely on hidden globals.
* The system **shall** raise `ValueError` on invalid shapes, mismatched dimensions, or invalid normalization when required.
* Tests **shall** run CPU-only and **shall** complete in ≤ 60 seconds on a reasonable laptop CPU.

## 6. Repository Layout

```
embkit/
  lib/
    models/
    index/
    query_ops/
    graph/
    safety/
    calibrate/
    eval/
    utils/
  experiments/
    configs/
    runs/
  notebooks/
    dashboards.ipynb
  tests/
    test_query_ops.py
    test_index.py
    test_graph.py
    test_calibration.py
    test_safety.py
  data/
    tiny/
  Makefile
  requirements.txt
  README.md
  LICENSE
```

* The repository **shall** conform to this structure. Filenames **shall not** differ.

## 7. Data Contracts

* **Input corpus (JSONL)**: one item per line with fields:

  * `id: str` (required)
  * `text: str` (required)
  * `ts: int | null` epoch seconds (optional)
  * `group: str | null` (optional)
* **Precomputed embeddings**: `.npy` aligned to an `ids.txt` list; alignment **must** match one-to-one.
* The IO layer **shall** validate alignment and **shall** fail fast on mismatches.

## 8. Configuration Schema (YAML, Pydantic-validated)

* Top-level keys and allowed values:

  * `model.name: str`
  * `index.kind: "flatip" | "ivfpq"`
  * `index.params: { nlist:int, m:int, nbits:int, nprobe:int }` (required for `ivfpq`)
  * `query_ops: list[ dict ]` where each dict **shall** specify `op` and parameters
  * `graph: { enable: bool, k: int, alpha: float }`
  * `safety: { repellors: path|null, pii: { enable: bool } }`
  * `calibrate: { method: "temperature" | "isotonic", params: dict }`
  * `eval: { k: list[int], latency_trials: int }`
  * `paths: { corpus: str, output_dir: str }`
* The CLI **shall** fail fast if the config is invalid or incomplete.

## 9. Public APIs (Signatures and Behavior)

* Utilities:

  * `utils.l2n(x: np.ndarray, axis: int|None = 1) -> np.ndarray`

    * **Shall** L2-normalize over `axis`. **Shall** return float32. **Shall** raise `ValueError` on zero-norm vectors when strict mode is enabled.
* Index:

  * `index.FlatIP(d: int)`

    * `.add(D: np.ndarray, ids: list[str]) -> None` (**shall** L2-normalize internally)
    * `.search(q: np.ndarray, k: int) -> (list[str], np.ndarray)` (**shall** return ids and scores sorted desc)
  * `index.IVFPQ(d:int, nlist:int, m:int, nbits:int, nprobe:int)`

    * `.train_add(D: np.ndarray, ids: list[str]) -> None`
    * `.search(q: np.ndarray, k:int) -> (list[str], np.ndarray)` (**shall** perform IVF-PQ then exact residual re-rank over top-200)
* Query ops:

  * `query_ops.directional(q, v_dir, D, alpha: float) -> list[dict]`

    * **Shall** compute `cos(D, q + α v_dir)` and return top-k style result list when used by CLI.
  * `query_ops.contrastive(q, v_neg, D, lam: float) -> list[dict]`

    * **Shall** compute `cos(D,q) - λ cos(D,v_neg)`.
  * `query_ops.compose_and(qs: list[np.ndarray]) -> np.ndarray`

    * **Shall** sum then L2-normalize.
  * `query_ops.compose_or(scores: list[np.ndarray]) -> np.ndarray`

    * **Shall** elementwise max over aligned score arrays.
  * `query_ops.analogical(a,b,c) -> np.ndarray`

    * **Shall** return L2-normalized `b - a + c`.
  * `query_ops.mmr(q, D, k:int, lam: float) -> list[int]`

    * **Shall** return indices of selected items from top-50 by base cosine; **shall** be deterministic.
  * `query_ops.temporal(scores: np.ndarray, ages_days: np.ndarray, gamma: float) -> np.ndarray`

    * **Shall** apply `scores * exp(-γ * ages_days)`.
  * `query_ops.personalize(q, u, D, beta: float) -> np.ndarray`

    * **Shall** return `β cos(D,u) + cos(D,q)`.
  * `query_ops.cone_filter(q, D, cos_min: float) -> list[int]`

    * **Shall** return indices with `cos(D,q) ≥ cos_min`.
  * `query_ops.polytope_filter(D, constraints: list[tuple[np.ndarray,float]]) -> list[int]`

    * **Shall** return indices satisfying all half-space constraints.
  * `query_ops.mahalanobis_diag(u, v, w) -> float`

    * **Shall** return `sqrt( (u-v)^T diag(w) (u-v) )`; `w` **must** be nonnegative.
* Graph:

  * `graph.build_knn(D: np.ndarray, k:int) -> scipy.sparse.csr_matrix`

    * **Shall** build a kNN adjacency with nonnegative weights, no self-loops.
  * `graph.ppr(adj: csr_matrix, seed_scores: np.ndarray, alpha: float, iters:int) -> np.ndarray`

    * **Shall** implement power-iteration PPR; **shall** return a probability vector.
* Safety:

  * `safety.apply_repellors(scores: np.ndarray, D: np.ndarray, B: np.ndarray, lam: float) -> np.ndarray`

    * **Shall** subtract `λ * max_b cos(D, b)` from scores.
  * `safety.pii_contains(text: str) -> bool` and `safety.pii_redact(text: str) -> str`

    * **Shall** detect emails, SSNs, and common phones; **shall** redact deterministically.
* Calibration:

  * `calibrate.temperature_fit(y_true: np.ndarray, logits: np.ndarray) -> float`

    * **Shall** return optimal temperature `T > 0`.
  * `calibrate.temperature_apply(logits: np.ndarray, T: float) -> np.ndarray`

    * **Shall** apply calibrated mapping; **shall** preserve ordering.
* Evaluation:

  * `eval.metrics.compute_all(labels, rankings, times, ages) -> dict`

    * **Shall** compute Recall\@k, nDCG\@10, MRR, Diversity (1−mean cosine top-k), Freshness (time-weighted nDCG), ECE, Brier, and latency stats.
    * **Shall** emit JSONL rows with `{exp_id, metric, value, split, ts}`.

## 10. CLIs (Typer)

* `embkit index build --config PATH`

  * **Shall** build the configured index; **shall** persist FAISS files, id map, and metadata.
* `embkit search run --config PATH --query "..."`

  * **Shall** run search with configured pipeline; **shall** print top-5 and write full results.
* `embkit eval run --config PATH`

  * **Shall** execute evaluation and write metrics JSONL.
* CLIs **shall not** be interactive. CLIs **must** exit nonzero on validation errors.

## 11. Indexing Requirements

* **FlatIP** **shall** implement exact inner-product over L2-normalized vectors (i.e., cosine).
* **IVFPQ** **shall**:

  * Train on provided `D`; **shall** add all vectors.
  * Use `nprobe` from config.
  * Retrieve ≥200 candidates and **must** exact re-rank with full-precision vectors.
  * Persist and load deterministically.
* Index layers **shall** maintain an id↔row map; missing ids **must** fail fast.

## 12. Query Pipeline Requirements

* The pipeline **shall** support composing multiple ops in order defined by config.
* Each op **shall** document its input/outputs and **shall not** mutate shared state.
* MMR **shall** re-rank over the top-50 candidates from the prior step.
* Temporal decay **shall** use `ts` if available; otherwise **shall** treat age as 0.

## 13. Graph Fusion

* When `graph.enable` is true, the system **shall**:

  * Build a kNN graph with `graph.k` neighbors.
  * Run PPR with restart `graph.alpha`.
  * Fuse vector scores with PPR scores via linear combination specified in config (default equal weights).
* Graph routines **shall** run CPU-only and **shall** complete on the tiny dataset within test time.

## 14. Safety and Privacy

* If `safety.pii.enable` is true, snippets and logs **must** be redacted before writing to disk.
* If `safety.repellors` is provided, the pipeline **shall** load repellor vectors and **shall** apply the penalty before final ranking.
* The system **shall not** transmit text or embeddings over the network.

## 15. Calibration

* Temperature scaling **shall** reduce ECE on the provided synthetic split (acceptance test).
* Isotonic calibration (if chosen) **shall** be monotonic and **shall not** be extrapolated beyond training range without a clamp.

## 16. Evaluation and Logging

* The evaluation module **shall** compute and write metrics as JSONL under `experiments/runs/<exp_id>/metrics.jsonl`.
* The system **shall** log seed, versions, config echo, and timings to a plain-text log.
* Latency measurement **shall** report mean, p50, p95 over `eval.latency_trials`.

## 17. Performance and Quality Gates

* Tests **shall** pass in ≤ 60s CPU-only.
* On the tiny dataset:

  * FlatIP top-1 **shall** equal brute-force argmax.
  * IVFPQ Recall\@10 **shall** be ≥ 0.95 after re-rank.
  * Directional op **shall** shift rank per unit test.
  * Cone filter **shall** exclude wide-angle docs.
  * Polytope filter **shall** pass only items meeting all thresholds.
  * PPR **shall** rank neighbors of the seed above random nodes.
  * Temperature scaling **shall** reduce ECE on the synthetic split.
  * PII regex **shall** detect emails and SSNs; repellor penalty **shall** reduce scores for blocked topics.

## 18. Makefile Targets

* `make setup` **shall** install pinned requirements.
* `make quickstart` **shall** generate tiny data, build index, run one query, print top-5.
* `make test` **shall** run `pytest -q`.
* `make run EXP=demo` **shall** run evaluation with the demo config.

## 19. Coding Standards

* All public APIs **shall** have type hints and Google-style docstrings.
* The codebase **shall not** import seaborn or any networking libraries.
* Modules **shall not** use hidden module-level mutable state for core logic.
* Input validation **must** raise `ValueError` with clear messages.

## 20. Deliverables

* A working library under `embkit/lib` implementing modules listed.
* CLIs under `embkit/cli` using Typer.
* Tests under `embkit/tests` passing locally.
* Example configs under `experiments/configs/`.
* A demo run under `experiments/runs/demo/` with `metrics.jsonl` and logs.
* `README.md`, `requirements.txt`, `LICENSE`, `Makefile`.

## 21. Assumptions

* The license file **shall** exist; the specific license text **may** be MIT unless otherwise specified.
* The synthetic quickstart **shall** be sufficient to exercise all acceptance tests.
* Strict normalization checks **may** be enforced in query ops; index layers **shall** normalize defensively.

## 22. Out of Scope (v1)

* GPU acceleration, distributed indexing, external BM25 backends, cross-encoders, HNSW, networking, training pipelines.
* Full multi-vector retrieval beyond stubs.

## 23. Error Handling

* Configuration or IO errors **shall** produce a nonzero exit code and a concise error message.
* API misuse (shape, dtype, normalization) **must** raise `ValueError`.
* Missing files **shall** be reported with the offending path and required format.

## 24. Security

* Logs and artifacts **must not** contain unredacted PII when `pii.enable` is true.
* The system **shall not** write secrets, tokens, or environment dumps to logs.

---

### Appendix A — YAML Example (Normative)

```yaml
model:
  name: "dummy-encoder"
index:
  kind: "ivfpq"
  params: { nlist: 64, m: 8, nbits: 8, nprobe: 8 }
query_ops:
  - op: "directional"; alpha: 0.5; vector_path: "experiments/vectors/v_dir.npy"
  - op: "mmr"; lambda: 0.7; k: 10
graph:
  enable: true
  k: 10
  alpha: 0.15
safety:
  repellors: null
  pii: { enable: true }
calibrate:
  method: "temperature"
  params: { }
eval:
  k: [10, 100]
  latency_trials: 20
paths:
  corpus: "data/tiny/corpus.jsonl"
  output_dir: "experiments/runs/demo"
```

### Appendix B — Acceptance Tests (Normative)

* `test_index.py` **shall** verify FlatIP exactness and IVFPQ Recall\@10 ≥ 0.95.
* `test_query_ops.py` **shall** verify directional, cone, polytope behavior.
* `test_graph.py` **shall** verify PPR improves neighbor ranks.
* `test_calibration.py` **shall** verify ECE reduction post temperature scaling.
* `test_safety.py` **shall** verify PII regex and repellor penalty effects.

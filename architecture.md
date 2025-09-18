# An Advanced Vector-Embedding Operations Toolkit and Cookbook Architecture

## 1. Intent

Deliver An Advanced Vector-Embedding Operations Toolkit and Cookbook as a deterministic, CPU‑only toolkit for advanced vector retrieval. Provide core indexing, query algebra, graph fusion, safety, calibration, and evaluation with fast CLIs and unit tests. No network access. Float32 everywhere.

## 2. System boundary

* Inputs: YAML config, corpus JSONL, optional embeddings NPY, optional repellor vectors NPY.
* Outputs: FAISS index files, id maps, metadata, logs, metrics JSONL, quickstart demo data.
* Runtime: single process, single host, Python 3.11.

## 3. Technical pillars

* Cosine via inner product on L2‑normalized vectors.
* Exact baseline (FlatIP) and ANN (IVF‑PQ with exact re‑rank).
* Query algebra (directional, contrastive, AND/OR/NOT composition, analogical, MMR, temporal, personalization, cones, polytopes, diagonal Mahalanobis).
* Graph fusion (kNN + PPR).
* Safety (repellor penalty, PII redaction).
* Calibration (temperature, isotonic).
* Determinism (seed control, fixed n\_init, fixed threads).
* Evaluation (ranking, calibration, freshness, latency).

## 4. Data model

* Corpus JSONL: one record per line with `id`, `text`, optional `ts` (epoch seconds), optional `group`.
* Embeddings: float32 arrays. Docs matrix shape `(n, d)`. Query vector shape `(d,)`. Multi‑vector doc as list of `(m_i, d)`.
  When Hugging Face encoders are enabled we cache the raw matrix to `paths.embeddings` before L2 normalization.
* All embeddings L2‑normalized before any inner‑product search.
* Id map: contiguous zero‑based integer row ids ↔ external string ids.

## 5. Component map

* `lib/utils`: seeding, normalization, IO, timers, dtype checks, thread caps, config hashing.
* `lib/models`: encoder factory bridging the deterministic dummy encoder and CPU Hugging Face adapters with alias registry,
  batch helpers, and L2 normalization.
* `lib/index`: FlatIP exact; IVFPQ with residual exact re‑rank; artifact load/save; id mapping.
* `lib/query_ops`: scoring transforms, logical composition, filters, re‑rankers.
* `lib/graph`: kNN graph build (cosine), PPR power iteration, score fusion.
* `lib/safety`: repellor penalty, PII detection and redaction.
* `lib/calibrate`: temperature fit/apply; isotonic fit/apply.
* `lib/eval`: metrics and latency harness.
* `cli/`: Typer commands for index, search, eval.
* `experiments/`: configs, outputs, demo vectors.
* `tests/`: unit tests for acceptance gates.

## 6. Control flow (happy path)

1. CLI parses YAML. Pydantic validates schema and semantics. Fail fast on any invalid field.
2. Utils set seeds and cap BLAS threads. Record seed, versions, config hash.
3. IO loads corpus and optional embeddings. If absent, generate synthetic demo; when `provider: huggingface`, batch encode on
   CPU, persist embeddings, and backfill any missing ids.
4. Index layer builds or loads FAISS index and id map.
5. Search:

   * Encode or load query vector. Normalize.
   * Initial candidate retrieval (FlatIP or IVFPQ).
   * Apply query ops in configured order. Where ops change ranking, they work on candidate pools to bound cost.
   * Optional graph fusion via PPR.
   * Safety scoring penalty and PII redaction for outputs.
   * Optional calibration to map scores to probabilities.
6. Eval command runs metrics on provided labels or synthetic splits and writes JSONL.

## 7. Indexing choices

* Exact baseline: FAISS IndexFlatIP.

  * Rationale: simple, correct, strong baseline for tests and calibration.
* ANN: FAISS IVF‑PQ.

  * Coarse quantizer for partitions; product quantization for compression.
  * Deterministic training: fixed FAISS random seed, fixed training sample order.
  * Search: set `nprobe` from config. Retrieve at least 200 candidates. Re‑rank with exact float32 inner product on original vectors.
  * Rationale: common, well‑understood speed/recall trade‑off; residual re‑rank restores accuracy at top‑k.

## 8. Query algebra choices

* Directional: shift query along a direction vector to bias “more X”.
* Contrastive: penalize similarity to “not Y”.
* AND/OR/NOT: vector addition for AND, max fusion for OR, scaled subtraction for NOT.
* Analogical: classic `b − a + c` then normalize.
* MMR: trade relevance vs diversity. Applied over top‑50 to control cost.
* Temporal: exponential decay using ages in days.
* Personalization: linear blend with user vector.
* Cones: require cosine ≥ threshold to enforce angular locality.
* Polytopes: intersection of half‑spaces for multi‑constraint retrieval.
* Diagonal Mahalanobis: per‑dimension scaling weights for elliptical similarity.
* Rationale: covers common retrieval intents beyond nearest‑neighbor; each op is deterministic and vectorizable.

## 9. Graph fusion

* Build kNN graph on doc embeddings using cosine. Remove self‑loops. Store as CSR.
* Row‑normalize to a transition matrix surrogate. PPR via fixed‑iteration power method with restart `alpha`.
* Seed distribution from nonnegative, normalized base query scores.
* Fuse scores as weighted sum with vector scores. Weights from config.
* Rationale: leverages transitive relevance and local manifold structure.

## 10. Safety layer

* Repellor penalty: subtract λ times the maximum cosine similarity to any block vector.
* PII screening: regex patterns for emails, SSNs, common phone formats. Redact before logging or presenting.
* Rationale: reduce exposure to sensitive themes and PII without complex models; deterministic and auditable.

## 11. Calibration

* Temperature scaling: fit scalar T on a validation split to minimize NLL or ECE proxy; apply logistic squashing to produce calibrated probabilities.
* Isotonic regression: optional monotonic mapping on held‑out scores. Clamp at boundaries at inference.
* Rationale: improve interpretability and interoperability of scores with downstream decision rules.

## 12. Evaluation

* Retrieval: Recall\@k, nDCG\@10, MRR, Coverage\@k.
* Diversity: 1 − mean pairwise cosine in top‑k.
* Freshness: time‑weighted nDCG using exponential decay.
* Calibration: ECE (binned), Brier.
* Latency: mean, p50, p95 over warm runs on CPU.
* Outputs: JSONL rows with exp id, metric, value, split, timestamp.
* Rationale: balanced view of quality, diversity, freshness, and reliability.

## 13. Determinism tactics

* Seeds: Python, NumPy, scikit‑learn set at process start; FAISS RNG seeded; PyTorch seeded when Hugging Face encoders are
  instantiated.
* Threads: cap OpenMP/MKL/OpenBLAS to 1. Document the environment variables in logs. Force PyTorch to a single CPU thread.
* KMeans or clustering parameters set with fixed `n_init` and `random_state`.
* Any sampling order is fixed. Any shuffling is seeded.
* Rationale: reproducible runs for CI and acceptance tests.

## 14. Performance profile

* Data types: float32 only to halve memory and improve cache behavior.
* Candidate bounds: MMR on top‑50; residual re‑rank on top‑200; PPR with fixed iterations (e.g., 20).
* Vector ops use NumPy BLAS with single thread to stabilize timings.
* IO uses memory‑mapped NPY optionally for large arrays (future).
* Rationale: meet ≤60 s test budget and keep latency predictable.

## 15. Storage and artifacts

* Index files: FAISS binary files per index and a JSON metadata sidecar with dimensions, training params, and seed.
* Id map: text or NPY file with aligned external ids.
* Metadata: JSON record of versions, seeds, config hash, build timings.
* Logs: plain text. One file per run under `experiments/runs/<exp_id>/`.
* Metrics: JSONL under the same run directory.

## 16. Configuration and validation

* Pydantic models enforce types, ranges, and cross‑field constraints.
* Required keys: `model.name`, `index.kind`, `paths.corpus`, `paths.output_dir`.
* Optional Hugging Face knobs: `model.provider`, `model.batch_size`, `model.max_length`, `model.cache_dir`, `model.revision`,
  and `paths.embeddings`.
* Conditional requirements: `index.params` present for `ivfpq`.
* File existence checks occur before execution. Fail fast with clear messages.

## 17. CLIs

* `index build`: read config, prepare embeddings (reusing dummy vectors or encoding via Hugging Face), build index, persist
  artifacts and model metadata.
* `search run`: load index and vectors, instantiate encoder via factory, run pipeline, print top‑k, write results with redaction
  if enabled.
* `eval run`: execute metrics suite and latency harness, write JSONL.
* Non‑interactive, deterministic, nonzero exit on any validation failure.

## 18. Testing strategy

* Unit tests cover:

  * FlatIP vs brute‑force top‑1 equality.
  * IVF‑PQ Recall\@10 ≥ 0.95 on demo set after re‑rank.
  * Query ops: directional rank shift; cone exclusion; polytope inclusion; MMR diversity behavior.
  * Graph: PPR elevates neighbors of seed over random.
  * Calibration: ECE decreases after temperature scaling on synthetic split.
  * Safety: regex detects emails and SSNs; repellor penalty lowers target scores.
* Tests run CPU‑only in ≤60 s. Seeds fixed. No network calls.

## 19. Error handling

* Shape, dtype, and normalization validation at module boundaries. Raise `ValueError` with concise diagnostics.
* Config errors stop execution before any heavy work.
* Missing files and misalignment errors include offending path and expected schema.

## 20. Security and privacy posture

* No network IO. No telemetry.
* Redaction applied before persistence when enabled.
* Logs exclude raw PII and large payloads.
* Optional repellors reduce exposure to sensitive content.

## 21. Extensibility points

* Models: plug additional encoders or loaders in `lib/models` with the same embedding contracts.
* Index: add HNSW or other FAISS wrappers later behind `index.kind`.
* Query ops: add new ops as pure functions with docstrings and Pydantic‑validated parameters.
* Graph: swap kNN builder or add meta‑path variants without changing PPR API.
* Calibration: extend to Platt scaling or multi‑feature calibration.
* Safety: add learned detectors or more regexes without changing the penalty/redact interfaces.

## 22. Rationale for key choices

* FAISS chosen for robustness and speed on CPU with strong primitives and artifact persistence.
* Cosine on L2‑normalized vectors avoids norm bias and aligns with common embedding training.
* Residual re‑rank restores accuracy lost to PQ while keeping ANN latency low.
* PPR complements vector scores with cluster structure, improving recall on near‑misses.
* Temperature scaling provides simple, stable calibration that preserves ranking.
* Pydantic and Typer reduce boilerplate and enforce correctness at the edges.
* Float32 standardizes numerics and avoids cross‑platform drift seen with float64 BLAS differences.

## 23. Operational concerns

* Resource limits: single‑process memory footprint logged; candidate counts configurable to fit RAM.
* Concurrency: single‑thread numeric libraries for determinism; parallelism deferred to a future version.
* Observability: timestamps, stage timings, sizes, and seeds recorded; config hash ensures traceability.
* Backups: index artifacts and metadata are sufficient to restore a run.

## 24. Limitations (v1)

* No GPU, no distributed indexing, no external sparse baseline, no cross‑encoder re‑ranker.
* Multi‑vector retrieval only stubbed; late interaction not implemented.
* Deletions in FAISS handled by id skipping; true deletions require periodic rebuilds.
* Regex PII detection covers structured PII only; entities like names are out of scope.

## 25. Future work

* HNSW and IVF‑HNSW hybrids; adaptive `nprobe`.
* Multi‑vector late interaction and passage‑level rerankers.
* Learned fairness rerank with exposure constraints.
* Density‑aware novelty and drift alarms with dashboards.
* Mixed‑curvature spaces and attribute‑aware subspace routing.
* On‑device semantic cache with TTL and utility‑based eviction.

## 26. Acceptance alignment

* Quality gates map directly to tests and metrics.
* Freshness via temporal decay measured by time‑weighted nDCG.
* Safety gates verified by PII detection and repellor exposure reduction.
* Reproducibility proved by seeds, thread caps, and artifact metadata.

## 27. Makefile targets (behavioral description)

* `setup`: install pinned wheels in a clean environment.
* `quickstart`: synthesize demo data, build index, run a sample query, print top‑5, write results.
* `test`: run pytest quietly; exit nonzero on any failure.
* `run EXP=demo`: run evaluation with the named config and write metrics/logs.

## 28. Documentation plan

* `README.md`: quickstart, concepts, CLI examples, config schema summary, assumptions.
* `architecture.md`: this document.
* `requirements.md`: normative “shall/must” requirements (separate file).
* Inline docstrings for all public APIs.

This architecture balances correctness, speed, determinism, and extensibility under a CPU‑only constraint. It explains every technical choice and how each supports the acceptance criteria.

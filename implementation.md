# implementation.md — An Advanced Vector-Embedding Operations Toolkit and Cookbook

This document is a complete, copy‑paste implementation guide. It includes file layout, code for every file, and commands to run. No external network calls. CPU‑only. Float32 everywhere. Deterministic.

---

## 0) Prereqs

* Python 3.11
* OS packages not required. CPU only.

Create a clean directory and place the files below exactly as shown.

---

## 1) Repository layout

```
embkit/
  __init__.py
  cli/
    __init__.py
    config.py
    index.py
    search.py
    eval.py
  lib/
    __init__.py
    utils/
      __init__.py
      demo_data.py
    models/
      __init__.py
      dummy.py
    index/
      __init__.py
      flatip.py
      ivfpq.py
    query_ops/
      __init__.py
    graph/
      __init__.py
      knn.py
    safety/
      __init__.py
      pii.py
      repellors.py
    calibrate/
      __init__.py
      temperature.py
    eval/
      __init__.py
      metrics.py
  experiments/
    configs/
      demo.yaml
    runs/
  data/
    tiny/        # will be auto-generated if empty
Makefile
requirements.txt
README.md
```

---

## 2) Dependencies

**requirements.txt**

```txt
numpy==1.26.4
scipy==1.12.0
scikit-learn==1.4.2
faiss-cpu==1.8.0
networkx==3.2.1
pydantic==2.6.3
pyyaml==6.0.1
typer==0.12.3
pytest==7.4.4
tqdm==4.66.2
```

---

## 3) Makefile

**Makefile**

```make
PY=python3

setup:
	$(PY) -m pip install -r requirements.txt

quickstart:
	$(PY) -m embkit.cli.index build --config experiments/configs/demo.yaml
	$(PY) -m embkit.cli.search run --config experiments/configs/demo.yaml --query "example"

test:
	pytest -q

run:
	$(PY) -m embkit.cli.eval run --config experiments/configs/$(EXP).yaml
```

---

## 4) Config (YAML + validation)

**experiments/configs/demo.yaml**

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
seed: 42
top_k: 10
```

**embkit/cli/config.py**

```python
from __future__ import annotations
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, field_validator
import yaml, os

IndexKind = Literal["flatip", "ivfpq"]

class IndexParams(BaseModel):
    nlist: int
    m: int
    nbits: int
    nprobe: int

class ModelCfg(BaseModel):
    name: str

class IndexCfg(BaseModel):
    kind: IndexKind
    params: Optional[IndexParams] = None

class GraphCfg(BaseModel):
    enable: bool = False
    k: int = 10
    alpha: float = 0.15

class PiiCfg(BaseModel):
    enable: bool = True

class SafetyCfg(BaseModel):
    repellors: Optional[str] = None
    pii: PiiCfg = PiiCfg()

class CalibCfg(BaseModel):
    method: Literal["temperature", "isotonic"] = "temperature"
    params: Dict[str, Any] = {}

class EvalCfg(BaseModel):
    k: List[int] = [10, 100]
    latency_trials: int = 20

class PathsCfg(BaseModel):
    corpus: str
    output_dir: str

class Config(BaseModel):
    model: ModelCfg
    index: IndexCfg
    query_ops: List[Dict[str, Any]] = []
    graph: GraphCfg = GraphCfg()
    safety: SafetyCfg = SafetyCfg()
    calibrate: CalibCfg = CalibCfg()
    eval: EvalCfg = EvalCfg()
    paths: PathsCfg
    seed: int = 42
    top_k: int = 10

    @field_validator("index")
    @classmethod
    def _check_index(cls, v: IndexCfg):
        if v.kind == "ivfpq" and v.params is None:
            raise ValueError("index.params required for ivfpq")
        return v

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    cfg = Config(**y)
    if not os.path.exists(cfg.paths.corpus):
        os.makedirs(os.path.dirname(cfg.paths.corpus), exist_ok=True)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    return cfg
```

---

## 5) Utilities

**embkit/lib/**init**.py**

```python
# empty package marker
```

**embkit/lib/utils/**init**.py**

```python
from __future__ import annotations
import os, json, time, random
from typing import Iterable, List
import numpy as np

def set_num_threads(n: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    try:
        import faiss  # type: ignore
        faiss.omp_set_num_threads(n)
    except Exception:
        pass

def set_determinism(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

def l2n(x: np.ndarray, axis: int | None = 1, eps: float = 1e-12) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise ValueError("l2n expects numpy array")
    if axis is None:
        n = float(np.linalg.norm(x))
        if n < eps: raise ValueError("zero-norm vector")
        return (x / n).astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return (x / n).astype(np.float32, copy=False)

def assert_l2_normalized(x: np.ndarray, axis: int = 1, tol: float = 1e-3) -> None:
    norms = np.linalg.norm(x, axis=axis)
    if not np.allclose(norms, 1.0, atol=tol):
        raise ValueError("Embeddings must be L2-normalized")

class Timer:
    def __init__(self): self.t0 = time.perf_counter()
    def reset(self): self.t0 = time.perf_counter()
    def ms(self) -> float: return (time.perf_counter() - self.t0) * 1000.0

def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path): return []
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s: out.append(json.loads(s))
    return out

def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_npy(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr.astype(np.float32, copy=False))

def load_npy(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32, copy=False)
```

**embkit/lib/utils/demo\_data.py**

```python
from __future__ import annotations
import os, time
import numpy as np
from . import l2n, write_jsonl, save_npy, set_determinism

def generate_tiny(path_corpus: str, path_emb: str, n: int = 200, d: int = 32, seed: int = 42) -> None:
    set_determinism(seed)
    os.makedirs(os.path.dirname(path_corpus), exist_ok=True)
    ids = []
    rows = []
    t0 = int(time.time())
    # 2D latent then project to d
    rng = np.random.default_rng(seed)
    centers = np.array([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float32)
    X2 = []
    groups = []
    for i in range(n):
        c = centers[i % 2]
        x = c + rng.normal(0, 0.6, size=(2,))
        X2.append(x.astype(np.float32))
        groups.append("A" if i % 2 == 0 else "B")
        ids.append(f"doc_{i:04d}")
        rows.append({
            "id": ids[-1],
            "text": f"synthetic item {i} with latent {x[0]:.3f},{x[1]:.3f}",
            "ts": int(t0 - rng.integers(0, 90)*86400),
            "group": groups[-1]
        })
    X2 = np.stack(X2).astype(np.float32)
    # deterministic projection 2 -> d
    rng = np.random.default_rng(seed+1)
    W = rng.normal(0, 1.0, size=(2, d)).astype(np.float32)
    D = l2n(X2 @ W, axis=1)
    write_jsonl(path_corpus, rows)
    save_npy(path_emb, D)
    with open(os.path.join(os.path.dirname(path_corpus), "ids.txt"), "w", encoding="utf-8") as f:
        for s in ids: f.write(s+"\n")
```

---

## 6) Models

**embkit/lib/models/**init**.py**

```python
# package marker
```

**embkit/lib/models/dummy.py**

```python
from __future__ import annotations
import numpy as np
from ..utils import l2n, set_determinism

class DummyEncoder:
    """Deterministic hash 3-gram encoder for queries."""
    def __init__(self, d: int = 32, seed: int = 42):
        set_determinism(seed)
        self.d = int(d)
    def encode_query(self, text: str) -> np.ndarray:
        v = np.zeros(self.d, dtype=np.float32)
        t = f"##{text.lower()}##"
        for i in range(len(t)-2):
            tri = t[i:i+3]
            h = hash(tri) % self.d
            v[h] += 1.0
        return l2n(v, axis=None)
```

---

## 7) Indexes

**embkit/lib/index/**init**.py**

```python
# package marker
```

**embkit/lib/index/flatip.py**

```python
from __future__ import annotations
from typing import List, Tuple
import os, json
import numpy as np
import faiss  # type: ignore
from ..utils import l2n

class FlatIP:
    """Exact cosine via inner product on L2-normalized vectors."""
    def __init__(self, d: int):
        self.d = int(d)
        self.index = faiss.IndexFlatIP(self.d)
        self._ids: List[str] = []
        self._D: np.ndarray | None = None

    def add(self, D: np.ndarray, ids: List[str]) -> None:
        if D.dtype != np.float32: D = D.astype(np.float32)
        Dn = l2n(D, axis=1)
        if len(ids) != Dn.shape[0]:
            raise ValueError("ids length mismatch")
        self.index.add(Dn)
        self._ids = list(ids)
        self._D = Dn

    def search(self, q: np.ndarray, k: int) -> Tuple[List[str], np.ndarray]:
        qn = l2n(q.astype(np.float32), axis=None)[None, :]
        sims, I = self.index.search(qn, min(k, len(self._ids)))
        idx = I[0]
        idx = idx[idx >= 0]
        scores = sims[0][:len(idx)].astype(np.float32)
        return [self._ids[i] for i in idx], scores

    def save(self, dirpath: str) -> None:
        os.makedirs(dirpath, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dirpath, "flatip.faiss"))
        if self._D is not None:
            np.save(os.path.join(dirpath, "D_norm.npy"), self._D)
        with open(os.path.join(dirpath, "ids.txt"), "w", encoding="utf-8") as f:
            for s in self._ids: f.write(s+"\n")
        with open(os.path.join(dirpath, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"d": self.d, "kind": "flatip"}, f)

    @classmethod
    def load(cls, dirpath: str) -> "FlatIP":
        with open(os.path.join(dirpath, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(meta["d"])
        obj.index = faiss.read_index(os.path.join(dirpath, "flatip.faiss"))
        Dp = os.path.join(dirpath, "D_norm.npy")
        if os.path.exists(Dp):
            obj._D = np.load(Dp).astype(np.float32)
        with open(os.path.join(dirpath, "ids.txt"), "r", encoding="utf-8") as f:
            obj._ids = [ln.strip() for ln in f if ln.strip()]
        return obj
```

**embkit/lib/index/ivfpq.py**

```python
from __future__ import annotations
from typing import List, Tuple
import os, json
import numpy as np
import faiss  # type: ignore
from ..utils import l2n

class IVFPQ:
    """IVF-PQ ANN with exact residual re-rank of top-200."""
    def __init__(self, d: int, nlist: int, m: int, nbits: int, nprobe: int = 8):
        self.d, self.nlist, self.m, self.nbits, self.nprobe = int(d), int(nlist), int(m), int(nbits), int(nprobe)
        quantizer = faiss.IndexFlatIP(self.d)
        self.index = faiss.IndexIVFPQ(quantizer, self.d, self.nlist, self.m, self.nbits)
        self.index.nprobe = self.nprobe
        self._ids: List[str] = []
        self._D: np.ndarray | None = None

    def train_add(self, D: np.ndarray, ids: List[str]) -> None:
        if D.dtype != np.float32: D = D.astype(np.float32)
        Dn = l2n(D, axis=1)
        if not self.index.is_trained:
            self.index.train(Dn)
        self.index.add(Dn)
        self._D = Dn
        if len(ids) != Dn.shape[0]:
            raise ValueError("ids length mismatch")
        self._ids = list(ids)

    def search(self, q: np.ndarray, k: int) -> Tuple[List[str], np.ndarray]:
        if self._D is None: return [], np.array([], dtype=np.float32)
        qn = l2n(q.astype(np.float32), axis=None)[None, :]
        ncand = max(200, k * 5, 1)
        sims, I = self.index.search(qn, min(ncand, len(self._ids)))
        cand = I[0]
        cand = cand[cand >= 0]
        if cand.size == 0:
            return [], np.array([], dtype=np.float32)
        sims_exact = (self._D[cand] @ qn[0]).astype(np.float32)
        order = np.argsort(-sims_exact)[:k]
        chosen = cand[order]
        scores = sims_exact[order]
        return [self._ids[i] for i in chosen], scores

    def save(self, dirpath: str) -> None:
        os.makedirs(dirpath, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dirpath, "ivfpq.faiss"))
        if self._D is not None:
            np.save(os.path.join(dirpath, "D_norm.npy"), self._D)
        with open(os.path.join(dirpath, "ids.txt"), "w", encoding="utf-8") as f:
            for s in self._ids: f.write(s+"\n")
        with open(os.path.join(dirpath, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"d": self.d, "kind": "ivfpq", "nlist": self.nlist, "m": self.m, "nbits": self.nbits, "nprobe": self.nprobe},
                f
            )

    @classmethod
    def load(cls, dirpath: str) -> "IVFPQ":
        with open(os.path.join(dirpath, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(meta["d"], meta["nlist"], meta["m"], meta["nbits"], meta["nprobe"])
        obj.index = faiss.read_index(os.path.join(dirpath, "ivfpq.faiss"))
        Dp = os.path.join(dirpath, "D_norm.npy")
        if os.path.exists(Dp):
            import numpy as np
            obj._D = np.load(Dp).astype(np.float32)
        with open(os.path.join(dirpath, "ids.txt"), "r", encoding="utf-8") as f:
            obj._ids = [ln.strip() for ln in f if ln.strip()]
        return obj
```

---

## 8) Query operations

**embkit/lib/query\_ops/**init**.py**

```python
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from ..utils import l2n

def _cos_many(q: np.ndarray, D: np.ndarray) -> np.ndarray:
    qn = l2n(q.astype(np.float32), axis=None)
    Dn = l2n(D, axis=1)
    return (Dn @ qn).astype(np.float32)

def directional(q: np.ndarray, v_dir: np.ndarray, D: np.ndarray, alpha: float) -> List[Tuple[int, float]]:
    q_shift = l2n(q + float(alpha) * v_dir, axis=None)
    s = _cos_many(q_shift, D)
    order = np.argsort(-s)
    return [(int(i), float(s[i])) for i in order]

def contrastive(q: np.ndarray, v_neg: np.ndarray, D: np.ndarray, lam: float) -> List[Tuple[int, float]]:
    lam = float(lam)
    s = _cos_many(q, D) - lam * _cos_many(v_neg, D)
    order = np.argsort(-s)
    return [(int(i), float(s[i])) for i in order]

def compose_and(qs: List[np.ndarray]) -> np.ndarray:
    if len(qs) == 0: raise ValueError("compose_and needs at least one vector")
    v = np.zeros_like(qs[0], dtype=np.float32)
    for q in qs: v += q.astype(np.float32)
    return l2n(v, axis=None)

def compose_or(scores: List[np.ndarray]) -> np.ndarray:
    if len(scores) == 0: raise ValueError("compose_or needs at least one score array")
    out = scores[0].astype(np.float32)
    for s in scores[1:]:
        out = np.maximum(out, s.astype(np.float32))
    return out

def analogical(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return l2n(b - a + c, axis=None)

def mmr(q: np.ndarray, D: np.ndarray, k: int, lam: float) -> List[int]:
    k = int(k); lam = float(lam)
    sims_q = _cos_many(q, D)
    chosen: List[int] = []
    pool = np.arange(D.shape[0], dtype=int)
    for _ in range(min(k, len(pool))):
        rem = np.setdiff1d(pool, np.array(chosen, dtype=int), assume_unique=True)
        if len(chosen) == 0:
            i = int(rem[np.argmax(sims_q[rem])])
        else:
            Dn = l2n(D, axis=1)
            S = Dn[chosen]
            div = (Dn[rem] @ S.T).max(axis=1)
            score = lam * sims_q[rem] - (1 - lam) * div
            i = int(rem[np.argmax(score)])
        chosen.append(i)
    return chosen

def temporal(scores: np.ndarray, ages_days: np.ndarray, gamma: float) -> np.ndarray:
    return (scores.astype(np.float32) * np.exp(-float(gamma) * ages_days.astype(np.float32))).astype(np.float32)

def personalize(q: np.ndarray, u: np.ndarray, D: np.ndarray, beta: float) -> np.ndarray:
    return (_cos_many(q, D) + float(beta) * _cos_many(u, D)).astype(np.float32)

def cone_filter(q: np.ndarray, D: np.ndarray, cos_min: float) -> List[int]:
    s = _cos_many(q, D)
    return [int(i) for i in np.where(s >= float(cos_min))[0].tolist()]

def polytope_filter(D: np.ndarray, constraints: List[Tuple[np.ndarray, float]]) -> List[int]:
    Dn = l2n(D, axis=1)
    keep = np.ones(Dn.shape[0], dtype=bool)
    for v, thr in constraints:
        v = l2n(v.astype(np.float32), axis=None)
        keep &= (Dn @ v) >= float(thr)
    return [int(i) for i in np.where(keep)[0].tolist()]

def mahalanobis_diag(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    diff = (u.astype(np.float32) - v.astype(np.float32))
    if np.any(w < 0): raise ValueError("weights must be nonnegative")
    return float(np.sqrt(np.sum((diff * np.sqrt(w.astype(np.float32)))**2, dtype=np.float32)))
```

---

## 9) Graph

**embkit/lib/graph/**init**.py**

```python
# package marker
```

**embkit/lib/graph/knn.py**

```python
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from ..utils import l2n

def build_knn(D: np.ndarray, k: int) -> csr_matrix:
    """Build symmetric kNN graph with cosine weights, no self-loops."""
    Dn = l2n(D, axis=1)
    S = (Dn @ Dn.T).astype(np.float32)
    np.fill_diagonal(S, -np.inf)
    n = S.shape[0]
    k = int(min(max(1, k), n-1))
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    vals = S[np.arange(n)[:, None], idx].ravel()
    # make symmetric by max
    M = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    M = M.maximum(M.T)
    M.setdiag(0.0)
    M.eliminate_zeros()
    return M

def ppr(adj: csr_matrix, seed_scores: np.ndarray, alpha: float, iters: int = 20) -> np.ndarray:
    """Personalized PageRank with power iteration on row-normalized graph."""
    assert adj.shape[0] == adj.shape[1]
    n = adj.shape[0]
    # row-normalize
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg = np.maximum(deg, 1e-12)
    P = adj.multiply(1.0 / deg[:, None]).tocsr()
    s = seed_scores.astype(np.float32).copy()
    s = np.maximum(s, 0)
    s /= (s.sum() + 1e-12)
    r = s.copy()
    for _ in range(int(iters)):
        r = float(alpha) * s + (1.0 - float(alpha)) * (P.T @ r)
    return r.astype(np.float32)
```

---

## 10) Safety

**embkit/lib/safety/**init**.py**

```python
# package marker
```

**embkit/lib/safety/pii.py**

```python
from __future__ import annotations
import re

_RE_EMAIL = re.compile(r'\b[^@\s]+@[^@\s]+\.[^@\s]+\b')
_RE_SSN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
_RE_PHONE = re.compile(r'\b(?:\+?\d[\d\-\s\(\)]{7,}\d)\b')

def pii_contains(text: str) -> bool:
    return any(r.search(text) for r in (_RE_EMAIL, _RE_SSN, _RE_PHONE))

def pii_redact(text: str) -> str:
    t = _RE_EMAIL.sub("[REDACTED_EMAIL]", text)
    t = _RE_SSN.sub("[REDACTED_SSN]", t)
    t = _RE_PHONE.sub("[REDACTED_PHONE]", t)
    return t
```

**embkit/lib/safety/repellors.py**

```python
from __future__ import annotations
import numpy as np
from ..utils import l2n

def apply_repellors(scores: np.ndarray, D: np.ndarray, B: np.ndarray, lam: float) -> np.ndarray:
    """Subtract λ * max_b cos(D, b) from scores."""
    if B.size == 0: return scores.astype(np.float32)
    Dn = l2n(D, axis=1)
    Bn = l2n(B, axis=1)
    pen = (Dn @ Bn.T).max(axis=1).astype(np.float32)
    return (scores.astype(np.float32) - float(lam) * pen).astype(np.float32)
```

---

## 11) Calibration

**embkit/lib/calibrate/**init**.py**

```python
# package marker
```

**embkit/lib/calibrate/temperature.py**

```python
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def temperature_fit(y_true: np.ndarray, logits: np.ndarray) -> float:
    """Fit T>0 to minimize NLL of sigmoid(logits/T)."""
    y = y_true.astype(np.float64).ravel()
    z = logits.astype(np.float64).ravel()
    if y.shape != z.shape: raise ValueError("y_true and logits must align")

    def nll(T: float) -> float:
        T = max(T, 1e-6)
        p = _sigmoid(z / T)
        # avoid log(0)
        p = np.clip(p, 1e-6, 1-1e-6)
        return float(-(y * np.log(p) + (1-y) * np.log(1-p)).mean())

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)

def temperature_apply(logits: np.ndarray, T: float) -> np.ndarray:
    return _sigmoid(logits.astype(np.float64) / float(T)).astype(np.float32)
```

---

## 12) Evaluation

**embkit/lib/eval/**init**.py**

```python
# package marker
```

**embkit/lib/eval/metrics.py**

```python
from __future__ import annotations
import time
import numpy as np
from typing import Dict, List, Iterable

def recall_at_k(rels: List[int], ranked: List[int], k: int) -> float:
    if k <= 0: return 0.0
    topk = set(ranked[:k])
    hit = sum(1 for r in rels if r in topk)
    return float(hit / max(1, len(rels)))

def dcg_at_k(gains: List[int], k: int) -> float:
    g = np.array(gains[:k], dtype=np.float32)
    if g.size == 0: return 0.0
    discounts = 1.0 / np.log2(np.arange(2, g.size + 2))
    return float(np.sum(g * discounts))

def ndcg_at_k(ranked_rels: List[int], k: int = 10) -> float:
    dcg = dcg_at_k(ranked_rels, k)
    ideal = dcg_at_k(sorted(ranked_rels, reverse=True), k)
    return float(dcg / ideal) if ideal > 0 else 0.0

def mrr(ranked: List[int], rel_set: set[int]) -> float:
    for i, d in enumerate(ranked, start=1):
        if d in rel_set: return 1.0 / i
    return 0.0

def diversity_cosine(D_top: np.ndarray) -> float:
    if D_top.shape[0] < 2: return 0.0
    Dn = D_top / np.linalg.norm(D_top, axis=1, keepdims=True).clip(min=1e-12)
    S = Dn @ Dn.T
    n = D_top.shape[0]
    upper = S[np.triu_indices(n, k=1)]
    return float(1.0 - float(np.mean(upper))) if upper.size else 0.0

def time_weighted_ndcg(rels: List[int], ages_days: List[float], k: int, gamma: float) -> float:
    # gains times exp(-gamma * age)
    gains = np.array(rels[:k], dtype=np.float32) * np.exp(-gamma * np.array(ages_days[:k], dtype=np.float32))
    ideal = np.sort(gains)[::-1]
    discounts = 1.0 / np.log2(np.arange(2, min(k, gains.size) + 2))
    dcg = float(np.sum(gains[:discounts.size] * discounts))
    idcg = float(np.sum(ideal[:discounts.size] * discounts))
    return float(dcg / idcg) if idcg > 0 else 0.0

def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = y_true.astype(np.float32).ravel()
    pr = p.astype(np.float32).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (pr >= bins[i]) & (pr < bins[i+1] if i+1 < len(bins)-1 else pr <= bins[i+1])
        if m.sum() == 0: continue
        conf = float(pr[m].mean())
        acc = float(y[m].mean())
        ece += (m.sum() / len(pr)) * abs(conf - acc)
    return float(ece)

def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    y = y_true.astype(np.float32).ravel()
    pr = p.astype(np.float32).ravel()
    return float(np.mean((pr - y) ** 2))

def compute_all(labels, rankings, times, ages) -> Dict[str, float]:
    # Minimal aggregator for demo; users can extend.
    # labels: dict[qid] -> set(relevant doc ids)
    # rankings: dict[qid] -> list[doc ids]
    # times: dict[qid] -> float latency ms
    # ages: dict[qid] -> list[age_days per doc]
    out: Dict[str, float] = {}
    rec10 = []
    nd10 = []
    mrrs = []
    lat = []
    for qid, ranked in rankings.items():
        rels = labels.get(qid, set())
        rec10.append(recall_at_k(list(rels), ranked, 10))
        rbin = [1 if d in rels else 0 for d in ranked]
        nd10.append(ndcg_at_k(rbin, 10))
        mrrs.append(mrr(ranked, rels))
        lat.append(float(times.get(qid, 0.0)))
    out["Recall@10"] = float(np.mean(rec10)) if rec10 else 0.0
    out["nDCG@10"] = float(np.mean(nd10)) if nd10 else 0.0
    out["MRR"] = float(np.mean(mrrs)) if mrrs else 0.0
    out["Latency.p50.ms"] = float(np.median(lat)) if lat else 0.0
    out["Latency.p95.ms"] = float(np.percentile(lat, 95)) if lat else 0.0
    return out
```

---

## 13) CLIs

**embkit/cli/**init**.py**

```python
# package marker
```

**embkit/cli/index.py**

```python
from __future__ import annotations
import os, json
import typer
import numpy as np
from .config import load_config
from ..lib.utils import set_determinism, set_num_threads, read_jsonl, load_npy
from ..lib.utils.demo_data import generate_tiny
from ..lib.index.flatip import FlatIP
from ..lib.index.ivfpq import IVFPQ

app = typer.Typer(no_args_is_help=True, help="embkit index CLI")

@app.command("build")
def build(config: str):
    cfg = load_config(config)
    set_num_threads(1)
    set_determinism(cfg.seed)

    corpus = read_jsonl(cfg.paths.corpus)
    emb_path = os.path.join(os.path.dirname(cfg.paths.corpus), "embeddings.npy")
    ids_path = os.path.join(os.path.dirname(cfg.paths.corpus), "ids.txt")

    if len(corpus) == 0 or not os.path.exists(emb_path):
        typer.echo("Generating tiny synthetic dataset...")
        generate_tiny(cfg.paths.corpus, emb_path, n=200, d=32, seed=cfg.seed)

    if not os.path.exists(ids_path):
        # derive ids from corpus
        with open(ids_path, "w", encoding="utf-8") as f:
            for r in read_jsonl(cfg.paths.corpus):
                f.write(r["id"] + "\n")

    ids = [ln.strip() for ln in open(ids_path, "r", encoding="utf-8") if ln.strip()]
    D = load_npy(emb_path)
    if D.shape[0] != len(ids):
        raise typer.Exit(code=2)

    out_dir = os.path.join(cfg.paths.output_dir, "index")
    os.makedirs(out_dir, exist_ok=True)

    if cfg.index.kind == "flatip":
        idx = FlatIP(D.shape[1])
        idx.add(D, ids)
        idx.save(out_dir)
    elif cfg.index.kind == "ivfpq":
        p = cfg.index.params
        idx = IVFPQ(D.shape[1], p.nlist, p.m, p.nbits, p.nprobe)
        idx.train_add(D, ids)
        idx.save(out_dir)
    else:
        raise typer.Exit(code=2)

    meta = {"seed": cfg.seed, "versions": {}, "config": os.path.abspath(config)}
    with open(os.path.join(cfg.paths.output_dir, "build_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    typer.echo("Index build complete.")

if __name__ == "__main__":
    app()
```

**embkit/cli/search.py**

```python
from __future__ import annotations
import os, json
import typer
import numpy as np
from .config import load_config
from ..lib.utils import set_determinism, set_num_threads, read_jsonl
from ..lib.models.dummy import DummyEncoder
from ..lib.index.flatip import FlatIP
from ..lib.index.ivfpq import IVFPQ
from ..lib.query_ops import directional, mmr, contrastive
from ..lib.safety.pii import pii_redact, pii_contains

app = typer.Typer(no_args_is_help=True, help="embkit search CLI")

def _load_index(kind: str, path: str):
    p = os.path.join(path, "index")
    if kind == "flatip": return FlatIP.load(p)
    if kind == "ivfpq": return IVFPQ.load(p)
    raise ValueError("unknown index kind")

@app.command("run")
def run(config: str, query: str, k: int = 10):
    cfg = load_config(config)
    set_num_threads(1); set_determinism(cfg.seed)
    idx = _load_index(cfg.index.kind, cfg.paths.output_dir)
    # embedding for query
    emb_dim = int(getattr(idx, "d", 32))
    enc = DummyEncoder(d=emb_dim, seed=cfg.seed)
    q = enc.encode_query(query)
    ids, scores = idx.search(q, k=max(k, 50))
    # optional ops on candidates if available
    # if directional provided, try to apply on full matrix if present
    if cfg.query_ops:
        # load doc matrix if saved
        Dp = os.path.join(cfg.paths.output_dir, "index", "D_norm.npy")
        D = np.load(Dp) if os.path.exists(Dp) else None
        if D is not None:
            for op in cfg.query_ops:
                if op.get("op") == "directional":
                    vecp = op.get("vector_path")
                    v = np.load(vecp) if (vecp and os.path.exists(vecp)) else np.ones(D.shape[1], dtype=np.float32)
                    ranked = directional(q, v, D, op.get("alpha", 0.5))
                    # remap to current candidates
                    cand_set = {ids[i]: i for i in range(len(ids))}
                    re_sc = []
                    for i, s in ranked:
                        if i < D.shape[0]:
                            doc_id = open(os.path.join(cfg.paths.output_dir, "index", "ids.txt"), "r", encoding="utf-8").read().splitlines()[i]
                            if doc_id in cand_set:
                                re_sc.append((cand_set[doc_id], float(s)))
                    if re_sc:
                        order = [i for i, _ in sorted(re_sc, key=lambda x: -x[1])]
                        ids = [ids[i] for i in order[:k]]
                        scores = np.array([scores[i] for i in order[:k]], dtype=np.float32)
                if op.get("op") == "mmr":
                    # simple rerank inside candidates if D present
                    if D is not None:
                        # map candidate ids to rows
                        id2row = {s.strip(): j for j, s in enumerate(open(os.path.join(cfg.paths.output_dir, "index", "ids.txt"), "r", encoding="utf-8").read().splitlines())}
                        rows = np.array([id2row[i] for i in ids], dtype=int)
                        chosen_idx = mmr(q, D[rows], k=min(k, len(rows)), lam=op.get("lambda", 0.7))
                        ids = [ids[i] for i in chosen_idx]
                        scores = np.array([scores[i] for i in chosen_idx], dtype=np.float32)
                if op.get("op") == "contrastive":
                    if D is not None:
                        vneg_path = op.get("vector_path")
                        vneg = np.load(vneg_path) if (vneg_path and os.path.exists(vneg_path)) else np.ones(D.shape[1], dtype=np.float32)
                        from ..lib.query_ops import contrastive as _contrastive
                        # score full then filter to k
                        ranked = _contrastive(q, vneg, D, op.get("lambda", 0.5))
                        order = [i for i, _ in ranked]
                        idlist = open(os.path.join(cfg.paths.output_dir, "index", "ids.txt"), "r", encoding="utf-8").read().splitlines()
                        ids = [idlist[i] for i in order[:k]]
                        scores = np.array([1.0]*len(ids), dtype=np.float32)

    # PII redaction on output text if enabled
    docs = {r["id"]: r for r in read_jsonl(cfg.paths.corpus)}
    for i, doc_id in enumerate(ids[:k]):
        text = docs.get(doc_id, {}).get("text", "")
        if cfg.safety.pii.enable and pii_contains(text):
            text = pii_redact(text)
        typer.echo(f"{i+1:2d}. {doc_id}  score={scores[i]:.4f}  text={text[:80]}")

    # write results
    outp = os.path.join(cfg.paths.output_dir, "search_results.jsonl")
    with open(outp, "w", encoding="utf-8") as f:
        for i, doc_id in enumerate(ids[:k]):
            f.write(json.dumps({"rank": i+1, "id": doc_id, "score": float(scores[i])}) + "\n")
    typer.echo(f"Wrote {outp}")

if __name__ == "__main__":
    app()
```

**embkit/cli/eval.py**

```python
from __future__ import annotations
import os, time, json
import typer
import numpy as np
from .config import load_config
from ..lib.utils import set_determinism, set_num_threads, read_jsonl
from ..lib.eval.metrics import compute_all, expected_calibration_error, brier_score
from ..lib.calibrate.temperature import temperature_fit, temperature_apply

app = typer.Typer(no_args_is_help=True, help="embkit eval CLI")

@app.command("run")
def run(config: str):
    cfg = load_config(config)
    set_num_threads(1); set_determinism(cfg.seed)

    # Dummy evaluation over the tiny dataset for demonstration:
    rows = read_jsonl(cfg.paths.corpus)
    qids = [f"q{i:03d}" for i in range(min(5, len(rows)))]
    rankings = {qid: [rows[i]["id"] for i in range(len(rows))][:10] for qid in qids}
    labels = {qids[0]: set([rows[0]["id"]]), qids[1]: set([rows[1]["id"]])}
    times = {qid: 5.0 for qid in qids}
    ages = {qid: [0.0]*10 for qid in qids}
    metrics = compute_all(labels, rankings, times, ages)

    # Calibration demo
    y = np.array([1,0,1,0,1,0], dtype=np.float32)
    logits = np.array([2.0, 1.0, 1.5, -0.5, 3.0, -1.0], dtype=np.float32)
    T = temperature_fit(y, logits)
    p_raw = 1 / (1 + np.exp(-logits))
    p_cal = temperature_apply(logits, T)
    ece_raw = expected_calibration_error(y, p_raw)
    ece_cal = expected_calibration_error(y, p_cal)

    out_dir = cfg.paths.output_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.jsonl"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(json.dumps({"exp_id": "demo", "metric": k, "value": v, "split": "demo", "ts": int(time.time())}) + "\n")
        f.write(json.dumps({"exp_id": "demo", "metric": "ECE.raw", "value": float(ece_raw), "split": "demo", "ts": int(time.time())}) + "\n")
        f.write(json.dumps({"exp_id": "demo", "metric": "ECE.cal", "value": float(ece_cal), "split": "demo", "ts": int(time.time())}) + "\n")
    typer.echo(f"Wrote {os.path.join(out_dir, 'metrics.jsonl')}")

if __name__ == "__main__":
    app()
```

---

## 14) README (minimal)

**README.md**

````md
# An Advanced Vector-Embedding Operations Toolkit and Cookbook

An Advanced Vector-Embedding Operations Toolkit and Cookbook is a CPU-only vector embedding toolkit with FAISS indexing, query ops, graph fusion, safety, calibration, and evaluation.

## Setup
```bash
make setup
````

## Quickstart

```bash
make quickstart
```

## Tests

```bash
make test
```

````

---

## 15) Unit tests

Create a `tests/` directory at repo root. The tests below are deterministic and run under 60 seconds CPU-only.

**tests/test_index.py**
```python
import numpy as np
from embkit.lib.utils.demo_data import generate_tiny
from embkit.lib.utils import load_npy, read_jsonl, set_determinism
from embkit.lib.index.flatip import FlatIP
from embkit.lib.index.ivfpq import IVFPQ

def test_flatip_exact_top1(tmp_path):
    set_determinism(123)
    corpus = tmp_path / "corpus.jsonl"
    embp = tmp_path / "embeddings.npy"
    generate_tiny(str(corpus), str(embp), n=200, d=32, seed=123)
    D = load_npy(str(embp))
    ids = [f"doc_{i:04d}" for i in range(D.shape[0])]

    idx = FlatIP(D.shape[1])
    idx.add(D, ids)

    # pick a true document as query
    q = D[17].copy()
    got_ids, _ = idx.search(q, k=1)
    assert got_ids[0] == ids[17]

def test_ivfpq_recall10(tmp_path):
    set_determinism(123)
    corpus = tmp_path / "corpus.jsonl"
    embp = tmp_path / "embeddings.npy"
    generate_tiny(str(corpus), str(embp), n=200, d=32, seed=123)
    D = load_npy(str(embp))
    ids = [f"doc_{i:04d}" for i in range(D.shape[0])]

    ivf = IVFPQ(D.shape[1], nlist=64, m=8, nbits=8, nprobe=8)
    ivf.train_add(D, ids)

    # Evaluate recall@10 vs exact for 10 random queries
    rng = np.random.default_rng(123)
    recall = []
    for _ in range(10):
        qi = int(rng.integers(0, D.shape[0]))
        q = D[qi]
        # exact
        sims = (D @ q).astype(np.float32)
        exact_top10 = set(np.argsort(-sims)[:10].tolist())
        got_ids, _ = ivf.search(q, k=10)
        got_rows = set([int(s.split("_")[1]) for s in got_ids])  # parse index from id
        hit = len(exact_top10 & got_rows) / 10.0
        recall.append(hit)
    assert np.mean(recall) >= 0.95
````

**tests/test\_query\_ops.py**

```python
import numpy as np
from embkit.lib.query_ops import directional, cone_filter, polytope_filter
from embkit.lib.utils import l2n

def test_directional_shifts_rank():
    q = l2n(np.array([1.0, 0.0], dtype=np.float32), axis=None)
    vdir = l2n(np.array([0.0, 1.0], dtype=np.float32), axis=None)
    D = l2n(np.array([[0.9, 0.1], [0.4, 0.9]], dtype=np.float32), axis=1)
    base = directional(q, np.zeros_like(vdir), D, alpha=0.0)
    with_dir = directional(q, vdir, D, alpha=0.8)
    assert base[0][0] == 0
    assert with_dir[0][0] == 1  # direction pushes doc1 up

def test_cone_excludes_wide_angle():
    q = np.array([1.0, 0.0], dtype=np.float32)
    D = l2n(np.array([[0.95, 0.1], [0.5, 0.5], [-0.9, 0.0]], dtype=np.float32), axis=1)
    keep = cone_filter(q, D, cos_min=0.8)
    assert keep == [0]  # only near q

def test_polytope_intersection():
    D = l2n(np.array([[0.9,0.9], [0.9,0.1], [0.1,0.9]], dtype=np.float32), axis=1)
    c1 = (np.array([1.0, 0.0], dtype=np.float32), 0.7)
    c2 = (np.array([0.0, 1.0], dtype=np.float32), 0.7)
    keep = polytope_filter(D, [c1, c2])
    assert keep == [0]
```

**tests/test\_graph.py**

```python
import numpy as np
from embkit.lib.graph.knn import build_knn, ppr
from embkit.lib.utils import l2n

def test_ppr_neighbors_rank_higher():
    # two clusters in 2D
    A = np.array([[1.0,0.0],[0.9,0.1],[0.95,-0.05]], dtype=np.float32)
    B = np.array([[-1.0,0.0],[-0.9,-0.1],[-0.95,0.05]], dtype=np.float32)
    D = l2n(np.vstack([A,B]), axis=1)
    adj = build_knn(D, k=2)
    seed = np.zeros(D.shape[0], dtype=np.float32); seed[0]=1.0  # first node in cluster A
    scores = ppr(adj, seed, alpha=0.15, iters=20)
    # neighbors in A should outrank random node in B
    top3 = np.argsort(-scores)[:3].tolist()
    assert set(top3) <= {0,1,2}
```

**tests/test\_calibration.py**

```python
import numpy as np
from embkit.lib.calibrate.temperature import temperature_fit, temperature_apply
from embkit.lib.eval.metrics import expected_calibration_error

def test_temperature_reduces_ece():
    y = np.array([1,0,1,0,1,0,1,0], dtype=np.float32)
    logits = np.array([3.0, 2.5, 2.0, -1.0, 4.0, -2.0, 1.5, -0.5], dtype=np.float32)
    p_raw = 1.0 / (1.0 + np.exp(-logits))
    ece_raw = expected_calibration_error(y, p_raw, n_bins=10)
    T = temperature_fit(y, logits)
    p_cal = temperature_apply(logits, T)
    ece_cal = expected_calibration_error(y, p_cal, n_bins=10)
    assert ece_cal <= ece_raw
```

**tests/test\_safety.py**

```python
import numpy as np
from embkit.lib.safety.pii import pii_contains, pii_redact
from embkit.lib.safety.repellors import apply_repellors
from embkit.lib.utils import l2n

def test_pii_regex():
    assert pii_contains("Contact: john.doe@example.com")
    assert pii_contains("SSN 123-45-6789")
    assert not pii_contains("No sensitive data")
    red = pii_redact("Mail me at jane@site.org, SSN 123-45-6789, phone +1 (212) 555-1212")
    assert "[REDACTED_EMAIL]" in red and "[REDACTED_SSN]" in red and "[REDACTED_PHONE]" in red

def test_repellor_penalty():
    D = l2n(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), axis=1)
    scores = np.array([0.9, 0.9], dtype=np.float32)
    B = l2n(np.array([[1.0, 0.0]], dtype=np.float32), axis=1)  # block first direction
    penalized = apply_repellors(scores, D, B, lam=0.5)
    assert penalized[0] < penalized[1]
```

---

## 16) How to run

Install and quickstart:

```bash
make setup
make quickstart
```

Run tests:

```bash
make test
```

Run eval on demo config:

```bash
make run EXP=demo
```

---

## 17) Implementation notes

* Determinism: the CLIs set single‑thread BLAS and seeds. FAISS uses the quantizer and training order deterministically on CPU.
* Normalization: all search relies on cosine via inner product. `l2n` is applied defensively in indexes and ops.
* IVF‑PQ: retrieves ≥200 candidates then exact re‑ranks with stored normalized matrix `D_norm.npy`.
* MMR: operates on candidate pools to keep latency bounded.
* PPR: fixed 20 iterations; restart `alpha` from config. Uses row‑normalized symmetric kNN.
* PII: regex covers emails, SSNs, common phones. Redaction occurs before output.
* Calibration: temperature scaling minimizes NLL; `temperature_apply` outputs calibrated probabilities; ordering preserved.

This document contains all code and guidance required to implement and run An Advanced Vector-Embedding Operations Toolkit and Cookbook end‑to‑end.

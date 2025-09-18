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

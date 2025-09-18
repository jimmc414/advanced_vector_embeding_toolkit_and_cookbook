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

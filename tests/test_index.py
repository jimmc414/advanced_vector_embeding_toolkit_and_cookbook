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

import numpy as np
from embkit.lib.utils.demo_data import generate_tiny
from embkit.lib.utils import load_npy, set_determinism
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

    # Generate larger dataset for more robust testing
    generate_tiny(str(corpus), str(embp), n=1000, d=32, seed=123)
    D = load_npy(str(embp))
    ids = [f"doc_{i:04d}" for i in range(D.shape[0])]

    ivf = IVFPQ(D.shape[1], nlist=32, m=8, nbits=8, nprobe=8)
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


def test_indices_support_multi_batch_add(tmp_path):
    set_determinism(321)
    corpus = tmp_path / "corpus.jsonl"
    embp = tmp_path / "embeddings.npy"
    generate_tiny(str(corpus), str(embp), n=200, d=32, seed=321)
    D = load_npy(str(embp))
    ids = [f"doc_{i:04d}" for i in range(D.shape[0])]
    split = D.shape[0] // 2

    flat = FlatIP(D.shape[1])
    flat.add(D[:split], ids[:split])
    flat.add(D[split:], ids[split:])

    empty = FlatIP(D.shape[1])
    empty_ids, empty_scores = empty.search(D[0], k=5)
    assert empty_ids == []
    assert empty_scores.size == 0

    got_first, _ = flat.search(D[1], k=1)
    got_second, _ = flat.search(D[-2], k=1)
    assert got_first[0] == ids[1]
    assert got_second[0] == ids[-2]

    ivf = IVFPQ(D.shape[1], nlist=32, m=8, nbits=8, nprobe=8)
    ivf.train_add(D[:split], ids[:split])
    ivf.train_add(D[split:], ids[split:])

    got_first_ivf, _ = ivf.search(D[1], k=1)
    got_second_ivf, _ = ivf.search(D[-2], k=1)
    assert got_first_ivf[0] == ids[1]
    assert got_second_ivf[0] == ids[-2]

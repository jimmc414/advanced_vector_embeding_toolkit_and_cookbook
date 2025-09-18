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

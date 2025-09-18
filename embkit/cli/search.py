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

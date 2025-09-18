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
from ..lib.safety.repellors import apply_repellors

app = typer.Typer(no_args_is_help=True, help="embkit search CLI")

def _load_index(kind: str, path: str):
    p = os.path.join(path, "index")
    if kind == "flatip":
        return FlatIP.load(p)
    if kind == "ivfpq":
        return IVFPQ.load(p)
    raise ValueError("unknown index kind")

@app.command("run")
def run(config: str, query: str, k: int = 10):
    cfg = load_config(config)
    set_num_threads(1)
    set_determinism(cfg.seed)
    idx = _load_index(cfg.index.kind, cfg.paths.output_dir)

    # embedding for query
    emb_dim = int(getattr(idx, "d", 32))
    enc = DummyEncoder(d=emb_dim, seed=cfg.seed)
    q = enc.encode_query(query)

    ids, scores = idx.search(q, k=max(k, 50))
    scores = np.array(scores, dtype=np.float32)

    index_dir = os.path.join(cfg.paths.output_dir, "index")
    ids_path = os.path.join(index_dir, "ids.txt")
    id_list = []
    if os.path.exists(ids_path):
        with open(ids_path, "r", encoding="utf-8") as f:
            id_list = [ln.strip() for ln in f if ln.strip()]
    id_to_row = {doc_id: i for i, doc_id in enumerate(id_list)}

    D = None
    D_path = os.path.join(index_dir, "D_norm.npy")
    if os.path.exists(D_path):
        D = np.load(D_path)

    # optional ops on candidates if available
    if cfg.query_ops and D is not None and len(id_list) == D.shape[0]:
        for op in cfg.query_ops:
            name = op.get("op")
            if name == "directional":
                vecp = op.get("vector_path")
                if vecp and os.path.exists(vecp):
                    v = np.load(vecp)
                else:
                    v = np.ones(D.shape[1], dtype=np.float32)
                ranked = directional(q, v, D, op.get("alpha", 0.5))
                cand_index = {doc_id: idx for idx, doc_id in enumerate(ids)}
                re_sc = []
                for row_idx, score in ranked:
                    if 0 <= row_idx < len(id_list):
                        doc_id = id_list[row_idx]
                        if doc_id in cand_index:
                            re_sc.append((cand_index[doc_id], float(score)))
                if re_sc:
                    re_sc.sort(key=lambda x: -x[1])
                    keep = [idx for idx, _ in re_sc[: min(k, len(re_sc))]]
                    ids = [ids[i] for i in keep]
                    scores = np.array([scores[i] for i in keep], dtype=np.float32)
            elif name == "mmr":
                cand_rows = [id_to_row.get(doc_id, -1) for doc_id in ids]
                valid = [i for i, row in enumerate(cand_rows) if row >= 0]
                if valid:
                    row_subset = np.array([cand_rows[i] for i in valid], dtype=int)
                    chosen = mmr(q, D[row_subset], k=min(k, len(row_subset)), lam=op.get("lambda", 0.7))
                    ids = [ids[valid[i]] for i in chosen]
                    scores = np.array([scores[valid[i]] for i in chosen], dtype=np.float32)
            elif name == "contrastive":
                vneg_path = op.get("vector_path")
                if vneg_path and os.path.exists(vneg_path):
                    vneg = np.load(vneg_path)
                else:
                    vneg = np.ones(D.shape[1], dtype=np.float32)
                ranked = contrastive(q, vneg, D, op.get("lambda", 0.5))
                subset = ranked[: min(k, len(ranked))]
                ids = [id_list[i] for i, _ in subset]
                scores = np.array([score for _, score in subset], dtype=np.float32)

    # apply repellor penalty if configured
    if cfg.safety.repellors and D is not None and id_to_row:
        rep_path = cfg.safety.repellors
        if os.path.exists(rep_path):
            repellors = np.load(rep_path)
            rows = [id_to_row.get(doc_id, -1) for doc_id in ids]
            valid = [i for i, row in enumerate(rows) if row >= 0]
            if valid:
                D_subset = D[np.array([rows[i] for i in valid], dtype=int)]
                penalized = apply_repellors(scores[valid], D_subset, repellors, lam=0.5)
                scores = scores.astype(np.float32)
                for idx_local, sc in zip(valid, penalized):
                    scores[idx_local] = sc
                order = np.argsort(-scores)[: min(k, len(scores))]
                ids = [ids[i] for i in order]
                scores = scores[order]

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
            f.write(json.dumps({"rank": i + 1, "id": doc_id, "score": float(scores[i])}) + "\n")
    typer.echo(f"Wrote {outp}")

if __name__ == "__main__":
    app()

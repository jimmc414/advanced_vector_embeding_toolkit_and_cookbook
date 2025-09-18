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

    if os.path.isabs(config):
        try:
            cfg_path = os.path.relpath(config, os.getcwd())
        except ValueError:
            cfg_path = os.path.abspath(config)
    else:
        cfg_path = os.path.normpath(config)

    meta = {"seed": cfg.seed, "versions": {}, "config": cfg_path}
    with open(os.path.join(cfg.paths.output_dir, "build_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    typer.echo("Index build complete.")

if __name__ == "__main__":
    app()

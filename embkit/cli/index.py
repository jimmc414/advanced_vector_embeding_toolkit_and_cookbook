from __future__ import annotations
import os, json
import typer
import numpy as np
from .config import load_config
from ..lib.models import create_encoder, resolve_model_name
from ..lib.utils import (
    load_npy,
    read_jsonl,
    save_npy,
    set_determinism,
    set_num_threads,
    write_jsonl,
)
from ..lib.utils.demo_data import generate_tiny
from ..lib.index.flatip import FlatIP
from ..lib.index.ivfpq import IVFPQ

app = typer.Typer(
    no_args_is_help=True,
    help="An Advanced Vector-Embedding Operations Toolkit and Cookbook index CLI",
)

@app.command("build")
def build(config: str):
    cfg = load_config(config)
    set_num_threads(1)
    set_determinism(cfg.seed)

    corpus = read_jsonl(cfg.paths.corpus)
    corpus_dir = os.path.dirname(cfg.paths.corpus)
    emb_path = cfg.paths.embeddings or (
        os.path.join(corpus_dir, "embeddings.npy") if corpus_dir else "embeddings.npy"
    )
    ids_path = os.path.join(corpus_dir, "ids.txt") if corpus_dir else "ids.txt"

    ids = []
    D = None

    if cfg.model.provider == "dummy":
        if len(corpus) == 0 or not os.path.exists(emb_path):
            typer.echo("Generating tiny synthetic dataset...")
            generate_tiny(cfg.paths.corpus, emb_path, n=200, d=32, seed=cfg.seed)
            corpus = read_jsonl(cfg.paths.corpus)
        if not os.path.exists(ids_path):
            with open(ids_path, "w", encoding="utf-8") as f:
                for r in corpus:
                    f.write(r["id"] + "\n")
        ids = [ln.strip() for ln in open(ids_path, "r", encoding="utf-8") if ln.strip()]
        D = load_npy(emb_path)
    elif cfg.model.provider == "huggingface":
        if len(corpus) == 0:
            typer.echo("Corpus is empty; cannot encode with Hugging Face model.")
            raise typer.Exit(code=2)

        texts = []
        ids = []
        updated = False
        for i, row in enumerate(corpus):
            texts.append(row.get("text", ""))
            doc_id = row.get("id") or f"doc_{i:05d}"
            ids.append(doc_id)
            if "id" not in row:
                row["id"] = doc_id
                updated = True

        D = None
        if os.path.exists(emb_path):
            try:
                cached = load_npy(emb_path)
            except Exception as exc:  # pragma: no cover - defensive logging
                typer.echo(f"Failed to load cached embeddings from {emb_path}: {exc}")
            else:
                if cached.ndim != 2:
                    typer.echo("Cached embeddings must be a 2D array; recomputing.")
                elif cached.shape[0] != len(ids):
                    typer.echo(
                        "Cached embeddings do not match corpus size; recomputing."
                    )
                else:
                    ids_match = True
                    if os.path.exists(ids_path):
                        try:
                            with open(ids_path, "r", encoding="utf-8") as f:
                                stored_ids = [ln.strip() for ln in f if ln.strip()]
                        except OSError as exc:  # pragma: no cover - defensive logging
                            typer.echo(
                                f"Failed to read stored IDs from {ids_path}: {exc}"
                            )
                            ids_match = False
                        else:
                            if len(stored_ids) != len(ids) or stored_ids != ids:
                                ids_match = False
                    if ids_match:
                        D = cached
                        typer.echo("Loaded cached embeddings from disk.")
                    else:
                        typer.echo("Cached embeddings IDs mismatch corpus; recomputing.")

        if D is None:
            encoder = create_encoder(cfg.model, seed=cfg.seed)
            D = encoder.encode_documents(texts)
            if D.shape[0] != len(ids):
                raise typer.Exit(code=2)
            save_npy(emb_path, D)
        if updated:
            write_jsonl(cfg.paths.corpus, corpus)
        with open(ids_path, "w", encoding="utf-8") as f:
            for doc_id in ids:
                f.write(doc_id + "\n")
    else:
        raise typer.Exit(code=2)

    if D is None:
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

    model_meta = {
        "provider": cfg.model.provider,
        "name": cfg.model.name,
    }
    if cfg.model.provider == "huggingface":
        model_meta["resolved"] = resolve_model_name(cfg.model.name)
        if cfg.model.revision:
            model_meta["revision"] = cfg.model.revision
    meta = {
        "seed": cfg.seed,
        "versions": {},
        "config": cfg_path,
        "model": model_meta,
        "paths": {"embeddings": emb_path},
    }
    with open(os.path.join(cfg.paths.output_dir, "build_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    typer.echo("Index build complete.")

if __name__ == "__main__":
    app()

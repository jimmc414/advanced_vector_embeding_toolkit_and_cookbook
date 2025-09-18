#!/usr/bin/env python3
"""Regenerate demo embeddings, FAISS index, and helper vectors.

The script rebuilds the synthetic demo assets that are too large or binary to
commit to source control. It prints SHA256 checksums for verification so users
can confirm their locally generated files match the documented values.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from embkit.cli.config import load_config
from embkit.cli.index import build as build_index
from embkit.lib.utils import l2n, read_jsonl
from embkit.lib.utils.demo_data import generate_tiny


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _generate_embeddings(corpus_path: Path, emb_path: Path, seed: int) -> None:
    original_corpus = None
    if corpus_path.exists():
        original_corpus = corpus_path.read_text(encoding="utf-8")
    generate_tiny(str(corpus_path), str(emb_path), n=200, d=32, seed=seed)
    if original_corpus is not None:
        corpus_path.write_text(original_corpus, encoding="utf-8")


def _compute_vectors(corpus_path: Path, emb_path: Path, out_dir: Path) -> Dict[str, Path]:
    rows = read_jsonl(str(corpus_path))
    if not rows:
        raise RuntimeError(f"Corpus at {corpus_path} is empty; run demo pipeline first.")
    D = np.load(str(emb_path)).astype(np.float32, copy=False)
    if D.shape[0] != len(rows):
        raise RuntimeError("Embedding matrix and corpus length mismatch.")

    groups: Dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        groups.setdefault(row.get("group", "default"), []).append(idx)

    group_a = groups.get("A", list(range(min(50, len(rows)))))
    group_b = groups.get("B", list(range(len(rows) // 2, len(rows))))
    if not group_a or not group_b:
        raise RuntimeError("Demo corpus missing expected groups 'A' and 'B'.")

    mu_a = D[np.array(group_a)].mean(axis=0)
    mu_b = D[np.array(group_b)].mean(axis=0)
    v_dir = l2n(mu_a - mu_b, axis=None)
    v_neg = l2n(mu_b - mu_a, axis=None)

    repellor_idx = np.array(group_b[: min(16, len(group_b))])
    repellors = l2n(D[repellor_idx], axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    paths["experiments/vectors/v_dir.npy"] = out_dir / "v_dir.npy"
    paths["experiments/vectors/v_neg.npy"] = out_dir / "v_neg.npy"
    paths["experiments/vectors/repellors_demo.npy"] = out_dir / "repellors_demo.npy"

    np.save(paths["experiments/vectors/v_dir.npy"], v_dir.astype(np.float32, copy=False))
    np.save(paths["experiments/vectors/v_neg.npy"], v_neg.astype(np.float32, copy=False))
    np.save(paths["experiments/vectors/repellors_demo.npy"], repellors.astype(np.float32, copy=False))
    return paths


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/configs/demo.yaml",
        help="Path to the demo YAML config to build (relative to repo root by default).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    os.chdir(ROOT)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    cfg = load_config(str(config_path))
    corpus_path = Path(cfg.paths.corpus)
    emb_path = corpus_path.with_name("embeddings.npy")

    _generate_embeddings(corpus_path, emb_path, seed=cfg.seed)
    build_index(str(config_path))

    vector_paths = _compute_vectors(corpus_path, emb_path, ROOT / "experiments" / "vectors")

    index_dir = Path(cfg.paths.output_dir) / "index"
    binary_targets = {
        "data/tiny/embeddings.npy": emb_path,
        "experiments/runs/demo/index/ivfpq.faiss": index_dir / "ivfpq.faiss",
        "experiments/runs/demo/index/D_norm.npy": index_dir / "D_norm.npy",
        "experiments/vectors/v_dir.npy": vector_paths["experiments/vectors/v_dir.npy"],
        "experiments/vectors/v_neg.npy": vector_paths["experiments/vectors/v_neg.npy"],
        "experiments/vectors/repellors_demo.npy": vector_paths["experiments/vectors/repellors_demo.npy"],
    }

    print("Generated demo artifacts:\n")
    for rel, path in binary_targets.items():
        print(f"  - {rel} -> {path}")

    print("\nSHA256 checksums:\n")
    for rel, path in sorted(binary_targets.items()):
        digest = _sha256(path)
        print(f"{digest}  {rel}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np

from embkit.lib.utils import read_jsonl, write_jsonl, save_npy, load_npy
from embkit.lib.utils.demo_data import generate_tiny


def test_write_jsonl_no_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = [{"a": 1}, {"b": 2}]

    write_jsonl("data.jsonl", rows)

    path = tmp_path / "data.jsonl"
    assert path.is_file()
    assert read_jsonl("data.jsonl") == rows


def test_save_npy_no_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    arr = np.random.rand(2, 3).astype(np.float32)

    save_npy("array.npy", arr)

    path = tmp_path / "array.npy"
    assert path.is_file()
    loaded = load_npy("array.npy")
    np.testing.assert_allclose(loaded, arr, rtol=1e-6, atol=1e-6)


def test_generate_tiny_no_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    generate_tiny("corpus.jsonl", "embeddings.npy", n=4, d=4, seed=0)

    assert (tmp_path / "corpus.jsonl").is_file()
    assert (tmp_path / "embeddings.npy").is_file()
    assert (tmp_path / "ids.txt").is_file()

    with open(tmp_path / "ids.txt", "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    assert len(ids) == 4

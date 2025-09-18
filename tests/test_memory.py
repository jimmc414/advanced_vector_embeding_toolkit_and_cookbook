import numpy as np

from embkit.lib.memory.compression import average_embedding, summarize_sentences
from embkit.lib.memory.knn import build_memory_prompt


def test_average_embedding_normalizes_mean():
    docs = [
        np.array([0.9, 0.1], dtype=np.float32),
        np.array([0.8, 0.2], dtype=np.float32),
        np.array([0.85, 0.15], dtype=np.float32),
    ]
    comp = average_embedding(docs)
    assert comp is not None
    norms = [np.dot(comp, d) / (np.linalg.norm(comp) * np.linalg.norm(d)) for d in docs]
    assert min(norms) > 0.98


def test_build_memory_prompt_appends_neighbors():
    query = "How to print Hello in Python?"
    neighbors = [{"summary": "Use print('Hello')"}]
    prompt = build_memory_prompt(query, neighbors)
    assert "[MEM]" in prompt
    assert "print('Hello')" in prompt


def test_summarize_sentences_returns_long_sentences():
    texts = ["Short. This is a much longer informative sentence about systems."]
    summary = summarize_sentences(texts, max_sentences=1)
    assert "informative" in summary

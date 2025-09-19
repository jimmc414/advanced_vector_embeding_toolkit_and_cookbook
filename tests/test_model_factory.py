import numpy as np
import pytest

from embkit.cli.config import ModelCfg
from embkit.lib.models import DummyEncoder, create_encoder, resolve_model_name


def test_create_encoder_returns_dummy_encoder(monkeypatch):
    cfg = ModelCfg(name="dummy-encoder")
    enc = create_encoder(cfg, seed=123, dimension=16)

    assert isinstance(enc, DummyEncoder)

    vecs = enc.encode_queries(["hello", "world"])
    assert vecs.shape == (2, 16)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_dummy_encoder_produces_stable_vectors_across_instances():
    text = "Deterministic hashing"
    enc1 = DummyEncoder(d=16, seed=777)
    enc2 = DummyEncoder(d=16, seed=777)

    vec1 = enc1.encode_document(text)
    vec2 = enc2.encode_document(text)

    np.testing.assert_array_equal(vec1, vec2)


def test_create_encoder_hf_uses_registry(monkeypatch):
    created = {}

    class StubEncoder:
        def __init__(self, cfg):
            created["cfg"] = cfg

        def encode_queries(self, texts):
            return np.zeros((len(texts), 4), dtype=np.float32)

        def encode_documents(self, texts):
            return np.zeros((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr("embkit.lib.models.HuggingFaceEncoder", StubEncoder)

    cfg = ModelCfg(provider="huggingface", name="gte-large", batch_size=8, max_length=128)
    enc = create_encoder(cfg, seed=7)

    assert isinstance(enc, StubEncoder)
    hf_cfg = created["cfg"]
    assert hf_cfg.name_or_path == resolve_model_name("gte-large")
    assert hf_cfg.batch_size == 8
    assert hf_cfg.max_length == 128
    assert hf_cfg.seed == 7


def test_create_encoder_hf_without_dependencies(monkeypatch):
    monkeypatch.setattr("embkit.lib.models.HuggingFaceEncoder", None)
    monkeypatch.setattr("embkit.lib.models.HuggingFaceModelConfig", None)

    cfg = ModelCfg(provider="huggingface", name="gte-large")

    with pytest.raises(RuntimeError):
        create_encoder(cfg, seed=0)

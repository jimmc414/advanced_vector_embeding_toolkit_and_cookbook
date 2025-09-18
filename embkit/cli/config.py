from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional

import os

import yaml
from pydantic import BaseModel, ValidationInfo, field_validator

from ..lib.models import MODEL_REGISTRY

IndexKind = Literal["flatip", "ivfpq"]


class IndexParams(BaseModel):
    nlist: int
    m: int
    nbits: int
    nprobe: int


class ModelCfg(BaseModel):
    provider: Literal["dummy", "huggingface"] = "dummy"
    name: str = "dummy-encoder"
    batch_size: int = 32
    max_length: int = 512
    cache_dir: Optional[str] = None
    revision: Optional[str] = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str, info: ValidationInfo) -> str:
        provider = info.data.get("provider", "dummy")
        if provider == "huggingface":
            if v in MODEL_REGISTRY or "/" in v:
                return v
            raise ValueError(f"Unknown Hugging Face model alias: {v}")
        return v

    @field_validator("batch_size", "max_length")
    @classmethod
    def _check_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("batch_size and max_length must be positive")
        return v

class IndexCfg(BaseModel):
    kind: IndexKind
    params: Optional[IndexParams] = None

class GraphCfg(BaseModel):
    enable: bool = False
    k: int = 10
    alpha: float = 0.15

class PiiCfg(BaseModel):
    enable: bool = True

class SafetyCfg(BaseModel):
    repellors: Optional[str] = None
    pii: PiiCfg = PiiCfg()

class CalibCfg(BaseModel):
    method: Literal["temperature", "isotonic"] = "temperature"
    params: Dict[str, Any] = {}

class EvalCfg(BaseModel):
    k: List[int] = [10, 100]
    latency_trials: int = 20

class PathsCfg(BaseModel):
    corpus: str
    output_dir: str
    embeddings: Optional[str] = None

class Config(BaseModel):
    model: ModelCfg
    index: IndexCfg
    query_ops: List[Dict[str, Any]] = []
    graph: GraphCfg = GraphCfg()
    safety: SafetyCfg = SafetyCfg()
    calibrate: CalibCfg = CalibCfg()
    eval: EvalCfg = EvalCfg()
    paths: PathsCfg
    seed: int = 42
    top_k: int = 10

    @field_validator("index")
    @classmethod
    def _check_index(cls, v: IndexCfg):
        if v.kind == "ivfpq" and v.params is None:
            raise ValueError("index.params required for ivfpq")
        return v

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    cfg = Config(**y)
    corpus_dir = os.path.dirname(cfg.paths.corpus)
    if corpus_dir and not os.path.exists(corpus_dir):
        os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    if cfg.paths.embeddings:
        emb_dir = os.path.dirname(cfg.paths.embeddings)
        if emb_dir and not os.path.exists(emb_dir):
            os.makedirs(emb_dir, exist_ok=True)
    return cfg

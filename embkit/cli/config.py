from __future__ import annotations
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, field_validator
import yaml, os

IndexKind = Literal["flatip", "ivfpq"]

class IndexParams(BaseModel):
    nlist: int
    m: int
    nbits: int
    nprobe: int

class ModelCfg(BaseModel):
    name: str

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
    return cfg

from __future__ import annotations
import os, time, json
import typer
import numpy as np
from .config import load_config
from ..lib.utils import set_determinism, set_num_threads, read_jsonl
from ..lib.eval.metrics import compute_all, expected_calibration_error, brier_score
from ..lib.calibrate.temperature import temperature_fit, temperature_apply

app = typer.Typer(no_args_is_help=True, help="embkit eval CLI")

@app.command("run")
def run(config: str):
    cfg = load_config(config)
    set_num_threads(1); set_determinism(cfg.seed)

    # Dummy evaluation over the tiny dataset for demonstration:
    rows = read_jsonl(cfg.paths.corpus)
    qids = [f"q{i:03d}" for i in range(min(5, len(rows)))]
    rankings = {qid: [rows[i]["id"] for i in range(len(rows))][:10] for qid in qids}
    labels = {qid: {rows[i]["id"]} for i, qid in enumerate(qids) if i < len(rows)}
    times = {qid: 5.0 for qid in qids}
    ages = {qid: [0.0]*10 for qid in qids}
    metrics = compute_all(labels, rankings, times, ages)

    # Calibration demo
    y = np.array([1,0,1,0,1,0], dtype=np.float32)
    logits = np.array([2.0, 1.0, 1.5, -0.5, 3.0, -1.0], dtype=np.float32)
    T = temperature_fit(y, logits)
    p_raw = 1 / (1 + np.exp(-logits))
    p_cal = temperature_apply(logits, T)
    ece_raw = expected_calibration_error(y, p_raw)
    ece_cal = expected_calibration_error(y, p_cal)

    out_dir = cfg.paths.output_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.jsonl"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(json.dumps({"exp_id": "demo", "metric": k, "value": v, "split": "demo", "ts": int(time.time())}) + "\n")
        f.write(json.dumps({"exp_id": "demo", "metric": "ECE.raw", "value": float(ece_raw), "split": "demo", "ts": int(time.time())}) + "\n")
        f.write(json.dumps({"exp_id": "demo", "metric": "ECE.cal", "value": float(ece_cal), "split": "demo", "ts": int(time.time())}) + "\n")
    typer.echo(f"Wrote {os.path.join(out_dir, 'metrics.jsonl')}")

if __name__ == "__main__":
    app()

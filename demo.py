#!/usr/bin/env python3
"""
Complete demo showing embkit capabilities.
Runs a full pipeline and displays results.
"""

import os
import json
from embkit.cli.index import build
from embkit.cli.search import run as search_run
from embkit.cli.eval import run as eval_run

def main():
    config_path = "experiments/configs/demo.yaml"

    print("=== Building Index ===")
    build(config_path)

    print("\n=== Running Search ===")
    search_run(config_path, "synthetic example", k=5)

    print("\n=== Running Evaluation ===")
    eval_run(config_path)

    print("\n=== Sample Results ===")
    metrics_path = "experiments/runs/demo/metrics.jsonl"
    if not os.path.exists(metrics_path):
        print("No metrics found at", metrics_path)
        return
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            metric = json.loads(line)
            print(f"{metric['metric']}: {metric['value']:.4f}")

if __name__ == "__main__":
    main()

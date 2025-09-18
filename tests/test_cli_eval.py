import json
import textwrap

import pytest

from embkit.cli.eval import run as eval_run


def _write_corpus(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.mark.parametrize("rows", [[], [{"id": "doc-0"}]])
def test_eval_cli_handles_small_corpus(tmp_path, rows):
    corpus_path = tmp_path / "corpus.jsonl"
    output_dir = tmp_path / "outputs"
    _write_corpus(corpus_path, rows)

    config_content = textwrap.dedent(
        f"""
        model:
          name: test-model
        index:
          kind: flatip
        paths:
          corpus: {corpus_path}
          output_dir: {output_dir}
        """
    ).strip()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)

    eval_run(str(config_path))

    metrics_path = output_dir / "metrics.jsonl"
    assert metrics_path.is_file()

    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    assert len(records) >= 7
    for record in records:
        assert record["metric"]
        assert isinstance(record["value"], (int, float))
        assert record["exp_id"] == "demo"
        assert record["split"] == "demo"

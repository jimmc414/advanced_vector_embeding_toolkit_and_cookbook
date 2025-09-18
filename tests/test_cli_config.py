import textwrap

import pytest

from embkit.cli.config import Config, ModelCfg, load_config


def test_load_config_allows_corpus_without_directory(tmp_path):
    output_dir = tmp_path / "outputs"
    config_content = textwrap.dedent(
        f"""
        model:
          name: dummy-encoder
        index:
          kind: flatip
        paths:
          corpus: corpus.txt
          output_dir: {output_dir}
        """
    ).strip()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)

    cfg = load_config(str(config_path))

    assert cfg.paths.corpus == "corpus.txt"
    assert cfg.paths.output_dir == str(output_dir)
    assert output_dir.is_dir()


def test_model_cfg_validates_hf_alias():
    cfg = Config(
        model=ModelCfg(provider="huggingface", name="e5-mistral"),
        index={"kind": "flatip"},
        paths={"corpus": "corpus.txt", "output_dir": "out"},
    )

    assert cfg.model.name == "e5-mistral"
    assert cfg.model.provider == "huggingface"


def test_model_cfg_rejects_unknown_hf_alias():
    with pytest.raises(ValueError):
        Config(
            model=ModelCfg(provider="huggingface", name="unknown-model"),
            index={"kind": "flatip"},
            paths={"corpus": "corpus.txt", "output_dir": "out"},
        )

import textwrap

from embkit.cli.config import load_config


def test_load_config_allows_corpus_without_directory(tmp_path):
    output_dir = tmp_path / "outputs"
    config_content = textwrap.dedent(
        f"""
        model:
          name: test
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

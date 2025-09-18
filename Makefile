PY=python3

setup:
$(PY) -m pip install -r requirements.txt

quickstart:
$(PY) -m embkit.cli.index build --config experiments/configs/demo.yaml
$(PY) -m embkit.cli.search run --config experiments/configs/demo.yaml --query "example"

test:
pytest -q

run:
$(PY) -m embkit.cli.eval run --config experiments/configs/$(EXP).yaml

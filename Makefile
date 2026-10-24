# Unix convenience targets. On Windows use: python scripts/tasks.py <target> --config ...
PY ?= python
CONFIG ?= configs/musique_200.yaml
TAG ?=

.PHONY: data index eval reproduce test lint

data:
	$(PY) scripts/tasks.py data --config $(CONFIG)

index:
	$(PY) scripts/tasks.py index --config $(CONFIG)

eval:
	$(PY) scripts/tasks.py eval --config $(CONFIG) --tag "$(TAG)"

reproduce:
	$(PY) scripts/tasks.py reproduce --config $(CONFIG) --tag "$(TAG)"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src scripts tests

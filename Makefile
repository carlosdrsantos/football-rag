PY := .venv/bin/python

.PHONY: install fetch parse chunk db lint test clean

install:
	python3 -m venv .venv
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q -e ".[dev]"

fetch:
	$(PY) -m lotg.ingest.fetch

parse:
	$(PY) -m lotg.ingest.parse

chunk:
	$(PY) -m lotg.ingest.chunk

db:
	docker compose up -d
	docker compose exec -T db pg_isready -U lotg -d lotg

lint:
	.venv/bin/ruff check src tests

test:
	.venv/bin/pytest -q

clean:
	rm -rf data/processed

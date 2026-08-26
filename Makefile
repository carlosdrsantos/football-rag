PY := .venv/bin/python

.PHONY: install fetch parse chunk db index search serve eval eval-answers lint test clean

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

index:
	$(PY) -m lotg.retrieval.build

search:
	$(PY) -m lotg.retrieval.search $(if $(R),-r $(R),) $(Q)

serve:
	.venv/bin/uvicorn lotg.service.app:app --host 0.0.0.0 --port 8000

eval:
	$(PY) -m lotg.evaluate

eval-answers:
	$(PY) -m lotg.evaluate_answers $(ARGS)

lint:
	.venv/bin/ruff check src tests

test:
	.venv/bin/pytest -q

clean:
	rm -rf data/processed

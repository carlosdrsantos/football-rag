"""Eval maths, and the schema contract the index depends on.

The pure functions are tested directly. The corpus check skips when
chunks.jsonl is absent, so the suite still runs on a clean checkout.
"""

import json
from pathlib import Path

import pytest

from lotg.evaluate import _majority_recall, _random_recall
from lotg.ingest.chunk import Chunk

CHUNKS = [{"law_number": 1}] * 10 + [{"law_number": 2}] * 90  # 100 chunks, 10 in Law 1


def test_random_recall_at_1_is_the_law_share():
    faqs = [{"law_number": 1}]
    assert _random_recall(CHUNKS, faqs, 1) == pytest.approx(0.10)
    faqs = [{"law_number": 2}]
    assert _random_recall(CHUNKS, faqs, 1) == pytest.approx(0.90)


def test_random_recall_rises_with_k():
    faqs = [{"law_number": 1}]
    scores = [_random_recall(CHUNKS, faqs, k) for k in (1, 3, 5, 10)]
    assert scores == sorted(scores)
    assert all(0 < s < 1 for s in scores)


def test_random_recall_is_certain_when_k_covers_the_corpus():
    faqs = [{"law_number": 1}]
    assert _random_recall(CHUNKS, faqs, len(CHUNKS)) == pytest.approx(1.0)


def test_random_recall_averages_over_queries():
    faqs = [{"law_number": 1}, {"law_number": 2}]
    assert _random_recall(CHUNKS, faqs, 1) == pytest.approx(0.50)


def test_majority_recall_picks_the_commonest_law():
    faqs = [{"law_number": 12}] * 3 + [{"law_number": 5}]
    recall, law = _majority_recall(faqs)
    assert law == 12
    assert recall == pytest.approx(0.75)


def test_chunks_jsonl_rehydrates_into_chunk():
    """build.py loads the index with Chunk(**row).

    A field added to Chunk but not written to the JSONL, or an extra key written
    to the JSONL, breaks the index build rather than any test of the chunker.
    """
    path = Path("data/processed/chunks.jsonl")
    if not path.exists():
        pytest.skip(f"{path} missing, run `make fetch parse chunk`")

    for line in path.open(encoding="utf-8"):
        Chunk(**json.loads(line))

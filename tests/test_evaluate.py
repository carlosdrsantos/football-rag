"""Eval maths, the gold set, and the schema contract the index depends on.

The pure functions are tested directly. The corpus checks skip when the parsed
data is absent, so the suite still runs on a clean checkout.
"""

import json
from pathlib import Path

import pytest

from lotg import gold
from lotg.evaluate import _majority_recall, _random_recall
from lotg.ingest.chunk import Chunk

CHUNKS = [{"law_number": 1}] * 10 + [{"law_number": 2}] * 90  # 100 chunks, 10 in Law 1
PROCESSED = Path("data/processed")


def law_sets(*laws: int | tuple[int, ...]) -> list[frozenset[int]]:
    return [frozenset(law if isinstance(law, tuple) else (law,)) for law in laws]


def test_random_recall_at_1_is_the_law_share():
    assert _random_recall(CHUNKS, law_sets(1), 1) == pytest.approx(0.10)
    assert _random_recall(CHUNKS, law_sets(2), 1) == pytest.approx(0.90)


def test_random_recall_counts_every_law_in_the_gold_set():
    # A blind draw hits {1, 2} always, because between them they are the corpus.
    assert _random_recall(CHUNKS, law_sets((1, 2)), 1) == pytest.approx(1.0)


def test_random_recall_rises_with_k():
    scores = [_random_recall(CHUNKS, law_sets(1), k) for k in (1, 3, 5, 10)]
    assert scores == sorted(scores)
    assert all(0 < s < 1 for s in scores)


def test_random_recall_is_certain_when_k_covers_the_corpus():
    assert _random_recall(CHUNKS, law_sets(1), len(CHUNKS)) == pytest.approx(1.0)


def test_random_recall_averages_over_queries():
    assert _random_recall(CHUNKS, law_sets(1, 2), 1) == pytest.approx(0.50)


def test_majority_recall_picks_the_commonest_law():
    recall, law = _majority_recall(law_sets(12, 12, 12, 5))
    assert law == 12
    assert recall == pytest.approx(0.75)


def test_majority_recall_credits_a_gold_set_containing_that_law():
    recall, law = _majority_recall(law_sets(12, (5, 12), 5))
    assert law == 12
    assert recall == pytest.approx(2 / 3)


FAQ_ROWS = [
    {"question": "Is the line part of the area?", "law_number": 1},
    {"question": "Is the line part of the area?", "law_number": 12},
    {"question": " Is the line part of the area? ", "law_number": 14},
    {"question": "When is the ball out of play?", "law_number": 9},
]


def test_cross_listed_rows_become_one_query():
    queries = gold.build(FAQ_ROWS)
    assert len(queries) == 2
    assert queries[0].laws == {1, 12, 14}
    assert queries[1].laws == {9}


def test_build_keeps_one_filed_law_per_row():
    # The single-label score is computed over these, so the count has to survive.
    assert sum(len(query.filed) for query in gold.build(FAQ_ROWS)) == len(FAQ_ROWS)


def test_real_eval_set_is_cross_listed():
    """Collapsing on question text is the whole fix, so it has to still apply."""
    path = PROCESSED / "faqs.jsonl"
    if not path.exists():
        pytest.skip(f"{path} missing, run `make fetch parse`")

    queries = gold.build([json.loads(line) for line in path.open(encoding="utf-8")])
    assert sum(len(q.laws) > 1 for q in queries) > 100


def test_chunks_jsonl_rehydrates_into_chunk():
    """build.py loads the index with Chunk(**row).

    A field added to Chunk but not written to the JSONL, or an extra key written
    to the JSONL, breaks the index build rather than any test of the chunker.
    """
    path = PROCESSED / "chunks.jsonl"
    if not path.exists():
        pytest.skip(f"{path} missing, run `make fetch parse chunk`")

    for line in path.open(encoding="utf-8"):
        Chunk(**json.loads(line))

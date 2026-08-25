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
NO_OVERRIDES = Path("does/not/exist.jsonl")


def test_cross_listed_rows_become_one_query():
    queries = gold.build(FAQ_ROWS, NO_OVERRIDES)
    assert len(queries) == 2
    assert queries[0].crosslisted == {1, 12, 14}
    assert queries[1].crosslisted == {9}


def test_build_keeps_one_filed_law_per_row():
    # The oldest score is computed over these, so the count has to survive.
    queries = gold.build(FAQ_ROWS, NO_OVERRIDES)
    assert sum(len(query.filed) for query in queries) == len(FAQ_ROWS)


def test_gold_defaults_to_the_filing(tmp_path):
    empty = tmp_path / "none.jsonl"
    empty.write_text("")
    for query in gold.build(FAQ_ROWS, empty):
        assert query.gold == query.crosslisted


def test_an_override_replaces_the_filing(tmp_path):
    path = tmp_path / "overrides.jsonl"
    target = gold.question_id("When is the ball out of play?")
    path.write_text(json.dumps({"id": target, "gold": [8, 9]}) + "\n")

    by_id = {q.id: q for q in gold.build(FAQ_ROWS, path)}
    assert by_id[target].gold == {8, 9}
    assert by_id[target].crosslisted == {9}, "the filing is kept for the older score"


def _real_queries() -> list:
    path = PROCESSED / "faqs.jsonl"
    if not path.exists():
        pytest.skip(f"{path} missing, run `make fetch parse`")
    return gold.build([json.loads(line) for line in path.open(encoding="utf-8")])


def test_real_eval_set_is_cross_listed():
    """Collapsing on question text is half the fix, so it has to still apply."""
    assert sum(len(q.crosslisted) > 1 for q in _real_queries()) > 100


def test_every_override_matches_a_real_question():
    """A mistyped id would silently relabel nothing and quietly weaken the key."""
    queries = _real_queries()
    known = {q.id for q in queries}
    unmatched = [row["id"] for row in _overrides() if row["id"] not in known]
    assert not unmatched, unmatched


def test_overrides_are_unique_and_name_real_laws():
    rows = _overrides()
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)), "an id listed twice means the last one silently wins"
    for row in rows:
        assert row["gold"], row
        assert all(1 <= law <= 17 for law in row["gold"]), row
        assert row["note"].strip(), f"{row['id']} needs a reason a human can audit"


def test_overrides_actually_change_the_key():
    relabelled = [q for q in _real_queries() if q.gold != q.crosslisted]
    assert len(relabelled) == len(_overrides())


def _overrides() -> list[dict]:
    path = Path("data/labels/overrides.jsonl")
    if not path.exists():
        pytest.skip(f"{path} missing")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


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

"""BM25 and rank fusion.

Both are pure functions over small inputs, so they are tested directly rather
than against the real index.
"""

import pytest

from lotg.retrieval import bm25
from lotg.retrieval.retrieve import fuse

DOCUMENTS = [
    "the ball is out of play when it leaves the field",
    "the ball is in play at all other times",
    "a defective ball is replaced and play restarts with a dropped ball",
]


@pytest.fixture
def index() -> bm25.BM25:
    return bm25.build(DOCUMENTS)


def test_a_rare_term_finds_its_document(index):
    ranked = index.search("defective", limit=3)
    assert ranked[0][0] == 2


def test_a_term_in_every_document_is_worth_nothing():
    # This is why there is no stopword list. IDF does the job on its own.
    index = bm25.build(["ball here", "ball there", "ball everywhere"])
    assert index.idf["ball"] == pytest.approx(0.0, abs=0.2)


def test_a_rare_term_outweighs_a_common_one(index):
    common = dict(index.search("ball", limit=3))
    rare = dict(index.search("defective ball", limit=3))
    assert rare[2] > common[2], "adding the discriminating term must move its document"


def test_terms_are_stemmed():
    index = bm25.build(["the player kicks the ball", "the referee watches"])
    assert index.search("kicked", limit=1)[0][0] == 0


def test_an_unknown_term_matches_nothing(index):
    assert index.search("substitution", limit=3) == []


def test_ties_break_on_index_so_a_run_is_reproducible():
    index = bm25.build(["offside", "offside", "offside"])
    assert [doc for doc, _ in index.search("offside", limit=3)] == [0, 1, 2]


def test_tokenize_drops_punctuation_and_case():
    assert bm25.tokenize("Throw-in, GOAL kick!") == bm25.tokenize("throw in goal kicks")


def test_fusion_prefers_what_both_rankings_like():
    dense = ["a", "b", "c"]
    lexical = ["a", "c", "b"]
    assert next(chunk for chunk, _ in fuse([dense, lexical])) == "a"


def test_fusion_can_rank_two_extremes_above_a_consistent_middle():
    """1/(k+1) + 1/(k+3) beats 2/(k+2), so RRF rewards being first somewhere.

    Worth pinning down, because it is the opposite of what averaging two scores
    would do, and it is why a chunk only BM25 likes can still reach the top.
    """
    ranked = [chunk for chunk, _ in fuse([["a", "b", "c"], ["c", "b", "a"]])]
    assert ranked == ["a", "c", "b"]


def test_fusion_reads_ranks_not_scores():
    """The whole point: cosine similarity and a BM25 score are not comparable."""
    assert fuse([["a", "b"]], rrf_k=60) == [("a", 1 / 61), ("b", 1 / 62)]


def test_fusion_keeps_a_chunk_only_one_ranking_found():
    fused = dict(fuse([["a"], ["b"]]))
    assert fused["a"] == fused["b"]


def test_fusion_ties_break_on_id():
    assert [chunk for chunk, _ in fuse([["b", "a"], ["a", "b"]])] == ["a", "b"]

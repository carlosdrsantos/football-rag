"""The generation layer, minus the model.

The prompt and the citation mapping are pure functions, so they are tested
directly. Nothing here calls Anthropic or needs Postgres.
"""

import pytest

from lotg.retrieval.store import Hit
from lotg.service.answer import Ruling, cited_hits, prompt


def hit(chunk_id: str, law: int) -> Hit:
    return Hit(chunk_id, law, f"Law {law} > {chunk_id}", f"text of {chunk_id}", "url", 1.0)


HITS = [hit("a", 12), hit("b", 5), hit("c", 3)]


def test_clauses_are_numbered_from_one():
    written = prompt("when is a coach sent off", HITS)
    assert "[1] Law 12 > a" in written
    assert "[3] Law 3 > c" in written
    assert "[0]" not in written


def test_prompt_carries_the_clause_text_and_the_question():
    written = prompt("who decides", HITS)
    assert "text of b" in written
    assert "Question: who decides" in written


def test_citations_map_back_to_the_right_chunks():
    ruling = Ruling(answer="Sent off [3].", cited=[3, 1], sufficient=True)
    assert [h.id for h in cited_hits(ruling, HITS)] == ["c", "a"]


def test_an_invented_citation_is_dropped():
    """The model cites by position, so a number past the end is the failure mode."""
    ruling = Ruling(answer="Per [9].", cited=[9, 2], sufficient=True)
    assert [h.id for h in cited_hits(ruling, HITS)] == ["b"]


@pytest.mark.parametrize("position", [0, -1, 4, 999])
def test_out_of_range_positions_never_reach_the_response(position):
    ruling = Ruling(answer="x", cited=[position], sufficient=True)
    assert cited_hits(ruling, HITS) == []


def test_a_repeated_citation_is_listed_once():
    ruling = Ruling(answer="[1] and again [1].", cited=[1, 1, 2], sufficient=True)
    assert [h.id for h in cited_hits(ruling, HITS)] == ["a", "b"]


def test_no_clauses_means_no_answer_and_no_call():
    from lotg.service.answer import answer

    ruling, cited = answer("anything", [])
    assert ruling.sufficient is False
    assert cited == []


ROWS = [
    # answered, cited a gold Law, and agreed
    {"sufficient": True, "cited_laws": [12], "gold_laws": [12], "retrieved_laws": [12, 5],
     "cited": ["a"], "agrees": True},
    # answered and agreed, but cited a Law outside the gold set alongside a good one
    {"sufficient": True, "cited_laws": [12, 7], "gold_laws": [12], "retrieved_laws": [12],
     "cited": ["a", "b"], "agrees": True},
    # abstained while holding the right Law, the only abstention worth fixing
    {"sufficient": False, "cited_laws": [], "gold_laws": [9], "retrieved_laws": [9, 8],
     "cited": [], "agrees": False},
    # abstained with nothing useful retrieved, which is the correct call
    {"sufficient": False, "cited_laws": [], "gold_laws": [2], "retrieved_laws": [12],
     "cited": [], "agrees": False},
]


def test_abstaining_and_still_reaching_the_right_ruling_is_counted():
    """A refusal that rules anyway hides a real failure behind a hedge."""
    from lotg.evaluate_answers import _score

    rows = [
        {"sufficient": False, "cited_laws": [], "gold_laws": [3], "retrieved_laws": [3],
         "cited": [], "agrees": True},
        {"sufficient": False, "cited_laws": [], "gold_laws": [3], "retrieved_laws": [3],
         "cited": [], "agrees": False},
    ]
    assert _score(rows)["abstained_but_ruled_correctly"] == pytest.approx(0.5)


def test_scoring_separates_the_two_kinds_of_abstention():
    from lotg.evaluate_answers import _score

    score = _score(ROWS)
    assert score["abstained"] == pytest.approx(0.5)
    assert score["abstained_with_the_answer_in_hand"] == pytest.approx(0.25)


def test_citing_a_gold_law_and_citing_only_gold_laws_differ():
    from lotg.evaluate_answers import _score

    score = _score(ROWS)
    assert score["cited_a_gold_law"] == pytest.approx(1.0)
    assert score["cited_only_gold_laws"] == pytest.approx(0.5)


def test_agreement_is_reported_over_all_questions_and_over_answered_ones():
    from lotg.evaluate_answers import _score

    score = _score(ROWS)
    assert score["agrees_with_ifab"] == pytest.approx(0.5)
    assert score["agrees_when_answered"] == pytest.approx(1.0)


def test_the_sample_is_stable_across_runs():
    from lotg.evaluate_answers import _sample

    queries = [f"q{i}" for i in range(60)]
    assert _sample(queries, 6, None) == _sample(queries, 6, None)
    assert _sample(queries, 6, None)[:3] == ["q0", "q6", "q12"]


def test_a_limit_thins_the_sample_instead_of_truncating_it():
    """The file is in Law order, so the first N questions are all Law 1 to 3."""
    from lotg.evaluate_answers import _sample

    queries = [f"q{i}" for i in range(100)]
    picked = _sample(queries, 1, 5)
    assert picked == ["q0", "q20", "q40", "q60", "q80"]


def test_a_limit_larger_than_the_sample_changes_nothing():
    from lotg.evaluate_answers import _sample

    queries = [f"q{i}" for i in range(10)]
    assert _sample(queries, 1, 99) == queries

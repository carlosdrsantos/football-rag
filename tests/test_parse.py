"""Parser regressions.

The unit tests run against a hand-written fixture and need no network. The
integration tests read the real parsed output and skip when it is absent.
"""

import json
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from lotg.ingest.parse import _parse_faqs, _parse_sections
from lotg.sources import LawPage

PAGE = LawPage(12, "Fouls and Misconduct", "fouls-and-misconduct")
PROCESSED = Path("data/processed")

# Mirrors the real markup: clause heading inside the collapse <button>, an
# <explanation> wrapping added and deleted wording, Q&A outside the <article>.
FIXTURE = """
<main>
  <article>
    <button type="button"><h2><span>12</span><span>.</span><span>4</span> Disciplinary action</h2></button>
    <p>A player is cautioned<explanation data-explanation="Why it changed">
       <addedexplanation>, except where a goal is scored</addedexplanation></explanation>.</p>
    <p>The referee may act<explanation data-explanation="Removed in this edition">
       <deletedexplanation> in an unsuccessful attempt to prevent a goal</deletedexplanation></explanation>.</p>
    <h3>Cautionable offences</h3>
    <ul><li>delaying the restart of play</li><li>dissent</li></ul>
  </article>
  <div class="QuestionAndAnswer__StyledQnAContainer-sc-1">
    <div class="QuestionAndAnswer__StyledQuestion-sc-2"><h2>Is handball always a penalty?</h2></div>
    <div class="QuestionAndAnswer__StyledAnswer-sc-3"><p>No, only deliberate handball.</p></div>
  </div>
</main>
"""


@pytest.fixture
def soup() -> BeautifulSoup:
    return BeautifulSoup(FIXTURE, "lxml")


@pytest.fixture
def section(soup) -> object:
    return _parse_sections(soup, PAGE)[0]


def test_clause_number_survives_the_collapse_button(section):
    # Dropping <button> as chrome once deleted every heading in the corpus.
    assert section.clause == "12.4"
    assert section.clause_title == "Disciplinary action"
    assert section.id == "law-12-12.4"


def test_repealed_wording_stays_out_of_the_markdown(section):
    assert "unsuccessful attempt to prevent a goal" not in section.markdown
    assert any("unsuccessful attempt" in t for t in section.removed_text)


def test_added_wording_is_kept_and_flagged(section):
    assert "except where a goal is scored" in section.markdown
    assert section.amended is True
    assert "Removed in this edition" in section.amendment_notes


def test_headings_and_lists_survive(section):
    assert "## 12.4 Disciplinary action" in section.markdown
    assert "### Cautionable offences" in section.markdown
    assert "- dissent" in section.markdown


def test_faqs_carry_their_answer(soup):
    faq = _parse_faqs(soup, PAGE)[0]
    assert faq.question == "Is handball always a penalty?"
    assert faq.answer == "No, only deliberate handball."
    assert faq.law_number == 12


def test_faqs_are_not_in_the_corpus(soup):
    corpus = "\n".join(s.markdown for s in _parse_sections(soup, PAGE))
    for faq in _parse_faqs(soup, PAGE):
        assert faq.question not in corpus
        assert faq.answer not in corpus


def _load(name: str) -> list[dict]:
    path = PROCESSED / name
    if not path.exists():
        pytest.skip(f"{path} missing, run `make fetch parse`")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def test_real_corpus_covers_all_seventeen_laws():
    assert {s["law_number"] for s in _load("sections.jsonl")} == set(range(1, 18))


def test_only_introductions_lack_a_clause_number():
    uncitable = [s for s in _load("sections.jsonl") if s["clause"] is None]
    assert all(s["clause_title"] == "Introduction" for s in uncitable), uncitable


def test_real_eval_set_does_not_leak_into_the_real_corpus():
    corpus = "\n".join(s["markdown"] for s in _load("sections.jsonl"))
    for faq in _load("faqs.jsonl"):
        assert faq["question"][:80] not in corpus

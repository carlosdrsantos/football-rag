"""What counts as a correct retrieval.

Two things are wrong with taking the Law a question was printed under as the
answer key, and they are fixed in that order.

IFAB publishes the same scenario on every Law page it touches. The 789 Q&A rows
are 595 distinct questions, and 171 of those are cross-listed under two or three
Laws:

    A defender recklessly fouls an attacker and the point of contact with the
    attacker's leg is on the penalty area line.

appears under Law 1, Law 12 and Law 14. Scoring that against one filed Law marks
retrieval wrong for returning a Law IFAB itself filed it under, and counts the
question three times while doing it. So a question is one query, and the filing
becomes a set.

That still leaves the questions IFAB filed only under the Law describing the
scenario while the ruling lives elsewhere. Law 9 asks what happens when the ball
hits the referee, and answers "dropped ball", which is Law 8.2. Every question
was read against its official answer and `overrides.jsonl` records the 23 where
the answer needs a Law the filing does not name, or names one the answer never
uses. Everything not listed there keeps IFAB's own filing.
"""

import json
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

OVERRIDES = Path("data/labels/overrides.jsonl")


@dataclass(frozen=True)
class Query:
    id: str
    question: str
    filed: tuple[int, ...]
    """The Law of every FAQ row that collapsed into this query, duplicates kept.

    Kept so the two older answer keys can be scored from the same retrieval
    pass: the set of these is IFAB's cross-listing, the tuple is the original
    one-row-one-Law key.
    """
    gold: frozenset[int]

    @property
    def crosslisted(self) -> frozenset[int]:
        return frozenset(self.filed)


def question_id(question: str) -> str:
    return sha1(question.strip().encode()).hexdigest()[:8]


def _overrides(path: Path) -> dict[str, list[int]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    return {row["id"]: row["gold"] for row in rows}


def build(faqs: list[dict], overrides: Path = OVERRIDES) -> list[Query]:
    """Collapse FAQ rows into one query per distinct question, first seen first."""
    reviewed = _overrides(overrides)
    filed: dict[str, list[int]] = {}
    for faq in faqs:
        filed.setdefault(faq["question"].strip(), []).append(faq["law_number"])

    queries = []
    for question, laws in filed.items():
        qid = question_id(question)
        gold = frozenset(reviewed.get(qid, laws))
        queries.append(Query(qid, question, tuple(laws), gold))
    return queries

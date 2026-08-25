"""What counts as a correct retrieval.

IFAB publishes the same scenario on every Law page it touches. The 789 Q&A rows
are 595 distinct questions, and 171 of those are cross-listed under two or three
Laws:

    A defender recklessly fouls an attacker and the point of contact with the
    attacker's leg is on the penalty area line.

appears under Law 1, Law 12 and Law 14. Scoring that question against one filed
Law marks retrieval wrong for returning a Law IFAB itself filed it under, and
counts the question three times while doing it. So a question is one query, and
its gold set is every Law it was filed under.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    question: str
    filed: tuple[int, ...]
    """The Law of every FAQ row that collapsed into this query, duplicates kept.

    The set of these is the gold label. The tuple is what the old single-label
    score is computed over, so both numbers come out of one retrieval pass.
    """

    @property
    def laws(self) -> frozenset[int]:
        return frozenset(self.filed)


def build(faqs: list[dict]) -> list[Query]:
    """Collapse FAQ rows into one query per distinct question, first seen first."""
    filed: dict[str, list[int]] = {}
    for faq in faqs:
        filed.setdefault(faq["question"].strip(), []).append(faq["law_number"])
    return [Query(question, tuple(laws)) for question, laws in filed.items()]

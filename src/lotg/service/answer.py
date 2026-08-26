"""Turn retrieved clauses into a ruling that cites them.

The failure this guards against is the one named at the top of the README: a
fluent answer citing the wrong clause. Three things push against it.

The model never sees the Laws, only the clauses retrieval returned, so it cannot
quote a rule that was not put in front of it. It cites by position, [1] to [k],
rather than reproducing clause numbers it could get subtly wrong, and the
positions are mapped back to real chunks here where a bad index is caught rather
than printed. And it is given an explicit way out: `sufficient: false` when the
clauses do not decide the question, which is the honest answer often enough that
not offering it would guarantee invention.
"""

import os
from functools import cache

import anthropic
from pydantic import BaseModel, Field

from lotg.retrieval.store import Hit

MODEL = os.environ.get("LOTG_MODEL", "claude-opus-5")
MAX_TOKENS = 4000

SYSTEM = """You are a football refereeing assistant answering from the IFAB Laws \
of the Game.

You will be given a question and a numbered list of clauses retrieved from the \
Laws. Those clauses are the only source you may use. You know a great deal about \
football, and none of it counts here: if a clause does not say it, you do not \
know it.

Rules:
- Answer only from the numbered clauses. Never rely on memory of the Laws.
- Put the citation marker for the clause that supports each statement directly \
after it, like [2]. A statement that no clause supports does not belong.
- Give the ruling first, in a sentence or two. Then the reason, if it needs one.
- Write for a referee who wants the decision, not an essay.
- If the clauses do not decide the question, set sufficient to false, say what is \
missing, and do not guess. Answering a question the clauses do not cover is a \
worse failure than admitting the gap."""


class Ruling(BaseModel):
    answer: str = Field(description="The ruling, with [n] markers citing clauses.")
    cited: list[int] = Field(description="Clause numbers used, as shown in the prompt.")
    sufficient: bool = Field(description="False when the clauses do not decide it.")


@cache
def client() -> anthropic.Anthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set, so /ask cannot answer")
    return anthropic.Anthropic()


def prompt(question: str, hits: list[Hit]) -> str:
    clauses = "\n\n".join(
        f"[{index}] {hit.breadcrumb}\n{hit.body}" for index, hit in enumerate(hits, start=1)
    )
    return f"Clauses:\n\n{clauses}\n\nQuestion: {question}"


def cited_hits(ruling: Ruling, hits: list[Hit]) -> list[Hit]:
    """Map the model's positions back to chunks, dropping any it invented."""
    seen: set[int] = set()
    ordered = []
    for position in ruling.cited:
        if 1 <= position <= len(hits) and position not in seen:
            seen.add(position)
            ordered.append(hits[position - 1])
    return ordered


def answer(
    question: str, hits: list[Hit], model: str | None = None
) -> tuple[Ruling, list[Hit]]:
    if not hits:
        return Ruling(answer="Nothing in the Laws matched that.", cited=[], sufficient=False), []

    response = client().messages.parse(
        model=model or MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt(question, hits)}],
        output_format=Ruling,
    )
    ruling = response.parsed_output
    return ruling, cited_hits(ruling, hits)

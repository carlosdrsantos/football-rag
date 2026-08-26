"""Measure the answers, not the retrieval.

Kept in its own file and its own output because it breaks two properties the
retrieval eval depends on: it costs money and it is not deterministic. Mixing
them would quietly cost `make eval` the thing that makes it worth trusting.

Most of what matters here still needs no judge. Whether the service abstained,
whether it cited a clause it was never shown, and whether the clause it cited
belongs to a gold Law are all set membership against labels that already exist.
Only agreement with IFAB's ruling needs a model, and that model is told to
compare two answers, never to grade retrieval.

The sample is every 6th question rather than a random draw, so a rerun measures
the same questions and two runs can be compared.

Retrieval runs first and sequentially, because a psycopg connection is not safe
to share across threads and 100 local searches cost about 35 seconds. The two API
calls per question are independent of every other question, so those run in a
thread pool. Sequentially they were the whole runtime.

The judge defaults to a different and cheaper model than the generator. Deciding
whether two answers reach the same ruling, with both in front of you, is the easy
half, and a model grading its own output rates it generously.
"""

import argparse
import itertools
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

from lotg import gold
from lotg.ingest import chunk as chunker
from lotg.retrieval import embedder, retrieve, store
from lotg.service import answer as answering

FAQS = Path("data/processed/faqs.jsonl")
RESULTS = Path("evals")
STRIDE = 6
TOP_K = 5
WORKERS = 8
JUDGE_MODEL = "claude-sonnet-5"

JUDGE_SYSTEM = """You are checking whether two answers to a refereeing question \
give the same ruling.

You will see the question, the official IFAB answer, and a candidate answer. \
Decide only whether the candidate reaches the same decision as the official one: \
the same restart, the same card, the same outcome.

Wording, length and ordering do not matter. Extra correct detail does not matter. \
A candidate that hedges but lands on the right decision agrees. A candidate that \
is fluent and well written but gives a different restart or a different card does \
not agree, and that is the case worth catching."""


class Judgement(BaseModel):
    agrees: bool
    note: str


def judge(
    question: str, official: str, candidate: str, model: str = JUDGE_MODEL
) -> Judgement:
    response = answering.client().messages.parse(
        model=model,
        max_tokens=1000,
        system=JUDGE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Official IFAB answer: {official}\n\n"
                    f"Candidate answer: {candidate}"
                ),
            }
        ],
        output_format=Judgement,
    )
    return response.parsed_output


def _progress(stage: str, done: int, total: int) -> None:
    """A long run with no output is indistinguishable from a hung one."""
    print(f"\r  {stage} {done}/{total}", end="", file=sys.stderr, flush=True)
    if done == total:
        print(file=sys.stderr)


def _sample(queries: list[gold.Query], stride: int, limit: int | None) -> list[gold.Query]:
    """Every Nth question, and a --limit that thins rather than truncates.

    faqs.jsonl is in Law order, so taking the first N of anything means Laws 1 to
    3 and nothing else. A six-question smoke run did exactly that and looked more
    representative than it was.
    """
    picked = queries[::stride]
    if not limit or limit >= len(picked):
        return picked
    step = len(picked) / limit
    return [picked[int(index * step)] for index in range(limit)]


def evaluate(
    stride: int = STRIDE,
    limit: int | None = None,
    k: int = TOP_K,
    workers: int = WORKERS,
    model: str | None = None,
    judge_model: str = JUDGE_MODEL,
) -> dict:
    faqs = [json.loads(line) for line in FAQS.open(encoding="utf-8")]
    official = {}
    for faq in faqs:
        official.setdefault(faq["question"].strip(), faq["answer"].strip())

    chunks = chunker.load()
    queries = _sample(gold.build(faqs), stride, limit)
    vectors = embedder.encode_queries([query.question for query in queries])

    retrieved = []
    with store.connect() as connection:
        lexical = retrieve.Lexical(chunks)
        stack = retrieve.Reranked(retrieve.Hybrid(retrieve.Dense(connection), lexical))
        for index, (query, vector) in enumerate(zip(queries, vectors), start=1):
            retrieved.append((query, stack.search(query.question, vector, k)))
            _progress("retrieving", index, len(queries))

    done = itertools.count(1)
    lock = threading.Lock()

    def work(pair):
        query, hits = pair
        ruling, cited = answering.answer(query.question, hits, model=model)
        verdict = judge(query.question, official[query.question], ruling.answer, judge_model)
        with lock:
            _progress("asking", next(done), len(retrieved))
        return {
            "id": query.id,
            "question": query.question,
            "gold_laws": sorted(query.gold),
            "retrieved_laws": [hit.law_number for hit in hits],
            "sufficient": ruling.sufficient,
            "cited_laws": [hit.law_number for hit in cited],
            "cited": [hit.id for hit in cited],
            "answer": ruling.answer,
            "agrees": verdict.agrees,
            "note": verdict.note,
        }

    # map keeps the input order, so the output does not depend on which call
    # finished first and two runs stay comparable.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(work, retrieved))
    print(file=sys.stderr)

    return {
        "model": model or answering.MODEL,
        "judge_model": judge_model,
        "k": k,
        "stride": stride,
        **_score(rows),
        "rows": rows,
    }


def _score(rows: list[dict]) -> dict:
    total = len(rows)
    answered = [row for row in rows if row["sufficient"]]
    abstained = [row for row in rows if not row["sufficient"]]

    # Abstaining is right when the clauses really did not hold the answer, and a
    # failure when they did. Only the second kind is worth fixing.
    wrongly_abstained = [
        row for row in abstained if set(row["retrieved_laws"]) & set(row["gold_laws"])
    ]
    # Claiming the clauses do not decide it and then reaching IFAB's ruling anyway.
    # The first run did this twice in six, which is how the broken abstention
    # contract was found: the flag was being used as a hedge, not a refusal.
    incoherent = [row for row in abstained if row["agrees"]]
    grounded = [row for row in answered if set(row["cited_laws"]) & set(row["gold_laws"])]
    clean = [row for row in answered if set(row["cited_laws"]) <= set(row["gold_laws"])]

    return {
        "questions": total,
        "answered": len(answered) / total,
        "abstained": len(abstained) / total,
        "abstained_with_the_answer_in_hand": len(wrongly_abstained) / total,
        "abstained_but_ruled_correctly": len(incoherent) / total,
        "cited_a_gold_law": len(grounded) / len(answered) if answered else 0.0,
        "cited_only_gold_laws": len(clean) / len(answered) if answered else 0.0,
        "agrees_with_ifab": sum(row["agrees"] for row in rows) / total,
        "agrees_when_answered": (
            sum(row["agrees"] for row in answered) / len(answered) if answered else 0.0
        ),
        "uncited_answers": sum(1 for row in answered if not row["cited"]),
        "per_law_questions": dict(
            sorted(Counter(law for row in rows for law in row["gold_laws"]).items())
        ),
    }


def report(result: dict) -> None:
    print(f"model      {result['model']}, top {result['k']} clauses")
    print(f"judge      {result['judge_model']}")
    print(f"questions  {result['questions']} (every {result['stride']}th)\n")
    print(f"  agrees with IFAB          {result['agrees_with_ifab']:.1%}")
    print(f"  agrees when it answered   {result['agrees_when_answered']:.1%}")
    print(f"  answered                  {result['answered']:.1%}")
    print(f"  abstained                 {result['abstained']:.1%}")
    print(f"    of which had the Law    {result['abstained_with_the_answer_in_hand']:.1%}")
    print(f"    but ruled anyway, right {result['abstained_but_ruled_correctly']:.1%}")
    print(f"  cited a gold Law          {result['cited_a_gold_law']:.1%}")
    print(f"  cited only gold Laws      {result['cited_only_gold_laws']:.1%}")
    print(f"  answered with no citation {result['uncited_answers']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=STRIDE, help="take every Nth question")
    parser.add_argument("--limit", type=int, help="stop after N of the sample")
    parser.add_argument("-k", type=int, default=TOP_K, help="clauses given to the model")
    parser.add_argument("--workers", type=int, default=WORKERS, help="parallel questions")
    parser.add_argument("--model", help=f"generator, default {answering.MODEL}")
    parser.add_argument("--judge", default=JUDGE_MODEL, help="model that compares answers")
    parser.add_argument("--save", default="answers", help="name under evals/")
    args = parser.parse_args()

    result = evaluate(
        stride=args.stride,
        limit=args.limit,
        k=args.k,
        workers=args.workers,
        model=args.model,
        judge_model=args.judge,
    )
    report(result)

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{args.save}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()

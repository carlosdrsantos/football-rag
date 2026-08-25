"""Measure retrieval against IFAB's own Q&A.

Each question is a refereeing scenario, and recall@k asks whether any chunk from
a Law IFAB filed it under came back. No annotation and no LLM judge, just set
membership over Law numbers.

Two baselines sit next to it, because recall@k alone is unreadable: always
answering from Law 12, and k blind draws, which hit a gold set covering n of N
chunks with probability 1 - C(N-n, k) / C(N, k).

The old single-label score is reported alongside. It scored each of the 789 FAQ
rows against the one Law that row was filed under, so it penalised cross-listed
questions. Both come out of the same retrieval pass, so the gap between them is
the cost of the old answer key and nothing else.
"""

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from math import comb
from pathlib import Path

from lotg import gold
from lotg.retrieval import embedder, store

FAQS = Path("data/processed/faqs.jsonl")
CHUNKS = Path("data/processed/chunks.jsonl")
RESULTS = Path("evals")
KS = (1, 3, 5, 10)

LawSets = list[frozenset[int]]


def _load(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing, run `make fetch parse chunk` first")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def _random_recall(chunks: list[dict], gold_sets: LawSets, k: int) -> float:
    """Expected recall@k if k chunks were drawn blind, computed exactly."""
    total = len(chunks)
    per_law = Counter(chunk["law_number"] for chunk in chunks)

    hits = 0.0
    for laws in gold_sets:
        relevant = sum(per_law[law] for law in laws)
        misses = comb(total - relevant, k) if total - relevant >= k else 0
        hits += 1 - misses / comb(total, k)
    return hits / len(gold_sets)


def _majority_recall(gold_sets: LawSets) -> tuple[float, int]:
    """Recall of always answering from whichever Law appears in the most gold sets."""
    law, _ = Counter(law for laws in gold_sets for law in laws).most_common(1)[0]
    return sum(law in laws for laws in gold_sets) / len(gold_sets), law


def _scores(chunks: list[dict], gold_sets: LawSets, correct: dict[int, int]) -> dict:
    return {
        "queries": len(gold_sets),
        "recall": {k: correct[k] / len(gold_sets) for k in KS},
        "baseline_random": {k: _random_recall(chunks, gold_sets, k) for k in KS},
    }


def evaluate(limit: int | None = None) -> dict:
    chunks = _load(CHUNKS)
    queries = gold.build(_load(FAQS))
    if limit:
        queries = queries[:limit]

    vectors = embedder.encode_queries([query.question for query in queries])

    top_k = max(KS)
    detail_k = min(KS)
    correct_at = dict.fromkeys(KS, 0)
    filed_correct_at = dict.fromkeys(KS, 0)
    per_law_total: dict[int, int] = defaultdict(int)
    per_law_correct: dict[int, int] = defaultdict(int)

    with store.connect() as connection:
        if store.count(connection) == 0:
            raise RuntimeError("index is empty, run `make index` first")

        for query, vector in zip(queries, vectors):
            laws = [hit.law_number for hit in store.search(connection, vector, top_k)]
            for k in KS:
                returned = set(laws[:k])
                correct_at[k] += bool(returned & query.laws)
                filed_correct_at[k] += sum(law in returned for law in query.filed)

            top = set(laws[:detail_k])
            for law in query.laws:
                per_law_total[law] += 1
                per_law_correct[law] += law in top

    gold_sets = [query.laws for query in queries]
    filed_sets = [frozenset([law]) for query in queries for law in query.filed]
    majority, majority_law = _majority_recall(gold_sets)
    return {
        "model": embedder.MODEL_NAME,
        "dimensions": embedder.dimensions(),
        "chunks": len(chunks),
        "cross_listed_queries": sum(len(laws) > 1 for laws in gold_sets),
        **_scores(chunks, gold_sets, correct_at),
        "single_label": _scores(chunks, filed_sets, filed_correct_at),
        "baseline_majority_law": {"law": majority_law, "recall@any": majority},
        "detail_k": detail_k,
        "per_law_recall_at_detail_k": {
            law: per_law_correct[law] / total for law, total in sorted(per_law_total.items())
        },
        "per_law_queries": dict(sorted(per_law_total.items())),
    }


def _table(scores: dict) -> Iterable[str]:
    yield f"{'k':>3}  {'recall':>7}  {'random':>7}  {'lift':>6}"
    for k in KS:
        got = scores["recall"][k]
        rand = scores["baseline_random"][k]
        yield f"{k:>3}  {got:>6.1%}  {rand:>6.1%}  {got / rand:>5.1f}x"


def report(result: dict) -> None:
    print(f"model    {result['model']} ({result['dimensions']}d)")
    print(f"corpus   {result['chunks']} chunks")
    print(
        f"queries  {result['queries']} distinct questions, "
        f"{result['cross_listed_queries']} filed under more than one Law\n"
    )

    print("\n".join(_table(result)))

    single = result["single_label"]
    print(f"\nunder the old single-label key ({single['queries']} FAQ rows):")
    print("\n".join(f"  {line}" for line in _table(single)))

    majority = result["baseline_majority_law"]
    print(
        f"\nalways answering from Law {majority['law']} would score "
        f"{majority['recall@any']:.1%} at any k"
    )

    print(f"\nrecall@{result['detail_k']} by Law (n = questions with that Law in the gold set):")
    for law, score in result["per_law_recall_at_detail_k"].items():
        n = result["per_law_queries"][law]
        bar = "#" * round(score * 20)
        print(f"  Law {law:>2}  n={n:>3}  {score:>6.1%}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="evaluate only the first N questions")
    parser.add_argument("--save", default="baseline", help="name under evals/")
    args = parser.parse_args()

    result = evaluate(limit=args.limit)
    report(result)

    if not args.limit:
        RESULTS.mkdir(exist_ok=True)
        path = RESULTS / f"{args.save}.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()

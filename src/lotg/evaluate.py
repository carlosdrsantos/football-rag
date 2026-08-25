"""Measure retrieval against IFAB's own Q&A.

Each FAQ is a refereeing scenario filed under a known Law, so recall@k here asks
whether any chunk from that Law came back. No annotation and no LLM judge.

Two baselines are reported next to it, because recall@k alone is unreadable:
always answering from Law 12 scores 37% (it holds 296 of the 789 questions), and
k blind draws hit a Law holding n of N chunks with probability
1 - C(N-n, k) / C(N, k).
"""

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

from lotg.retrieval import embedder, store

FAQS = Path("data/processed/faqs.jsonl")
CHUNKS = Path("data/processed/chunks.jsonl")
RESULTS = Path("evals")
KS = (1, 3, 5, 10)


def _load(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing, run `make fetch parse chunk` first")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def _random_recall(chunks: list[dict], faqs: list[dict], k: int) -> float:
    """Expected recall@k if k chunks were drawn blind, computed exactly."""
    total = len(chunks)
    per_law = defaultdict(int)
    for chunk in chunks:
        per_law[chunk["law_number"]] += 1

    hits = 0.0
    for faq in faqs:
        relevant = per_law[faq["law_number"]]
        misses = comb(total - relevant, k) if total - relevant >= k else 0
        hits += 1 - misses / comb(total, k)
    return hits / len(faqs)


def _majority_recall(faqs: list[dict]) -> tuple[float, int]:
    """Recall of always answering from whichever Law has the most questions."""
    per_law = defaultdict(int)
    for faq in faqs:
        per_law[faq["law_number"]] += 1
    law, count = max(per_law.items(), key=lambda item: item[1])
    return count / len(faqs), law


def evaluate(limit: int | None = None) -> dict:
    faqs = _load(FAQS)
    chunks = _load(CHUNKS)
    if limit:
        faqs = faqs[:limit]

    vectors = embedder.encode_queries([faq["question"] for faq in faqs])

    top_k = max(KS)
    detail_k = min(KS)
    correct_at = dict.fromkeys(KS, 0)
    per_law_total: dict[int, int] = defaultdict(int)
    per_law_correct: dict[int, int] = defaultdict(int)

    with store.connect() as connection:
        if store.count(connection) == 0:
            raise RuntimeError("index is empty, run `make index` first")

        for faq, vector in zip(faqs, vectors):
            hits = store.search(connection, vector, top_k)
            laws = [hit.law_number for hit in hits]
            per_law_total[faq["law_number"]] += 1
            for k in KS:
                if faq["law_number"] in laws[:k]:
                    correct_at[k] += 1
            if faq["law_number"] in laws[:detail_k]:
                per_law_correct[faq["law_number"]] += 1

    recall = {k: correct_at[k] / len(faqs) for k in KS}
    majority, majority_law = _majority_recall(faqs)
    return {
        "model": embedder.MODEL_NAME,
        "dimensions": embedder.dimensions(),
        "chunks": len(chunks),
        "queries": len(faqs),
        "recall": recall,
        "baseline_random": {k: _random_recall(chunks, faqs, k) for k in KS},
        "baseline_majority_law": {"law": majority_law, "recall@any": majority},
        "detail_k": detail_k,
        "per_law_recall_at_detail_k": {
            law: per_law_correct[law] / total for law, total in sorted(per_law_total.items())
        },
        "per_law_queries": dict(sorted(per_law_total.items())),
    }


def report(result: dict) -> None:
    print(f"model    {result['model']} ({result['dimensions']}d)")
    print(f"corpus   {result['chunks']} chunks")
    print(f"queries  {result['queries']} official FAQs\n")

    print(f"{'k':>3}  {'recall':>7}  {'random':>7}  {'lift':>6}")
    for k in KS:
        got = result["recall"][k]
        rand = result["baseline_random"][k]
        print(f"{k:>3}  {got:>6.1%}  {rand:>6.1%}  {got / rand:>5.1f}x")

    majority = result["baseline_majority_law"]
    print(
        f"\nalways answering from Law {majority['law']} would score "
        f"{majority['recall@any']:.1%} at any k"
    )

    print(f"\nrecall@{result['detail_k']} by Law (n = questions):")
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

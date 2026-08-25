"""Measure retrieval against IFAB's own Q&A.

Each question is a refereeing scenario, and recall@k asks whether any chunk from
a gold Law came back. No LLM judge and nothing to run at eval time but set
membership over Law numbers.

Every retriever answers the same questions in one pass, so the columns differ
only by the retriever. Two baselines sit next to them, because recall@k alone is
unreadable: always answering from Law 12, and k blind draws, which hit a gold set
covering n of N chunks with probability 1 - C(N-n, k) / C(N, k).

The two answer keys `gold.py` replaced are still scored, so a change to the key
can never be mistaken for a change to the retriever.
"""

import argparse
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

from lotg import gold
from lotg.ingest import chunk as chunker
from lotg.retrieval import embedder, retrieve, store

FAQS = Path("data/processed/faqs.jsonl")
RESULTS = Path("evals")
KS = (1, 3, 5, 10)
HEADLINE = "reranked"

Units = list[tuple[int, frozenset[int]]]
"""Each thing being scored: which query it came from, and its gold Laws."""


def _load(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing, run `make fetch parse chunk` first")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def _keys(queries: list[gold.Query]) -> dict[str, Units]:
    """The current answer key, and the two it replaced."""
    return {
        "gold": [(i, query.gold) for i, query in enumerate(queries)],
        "crosslisted": [(i, query.crosslisted) for i, query in enumerate(queries)],
        "filed": [
            (i, frozenset([law])) for i, query in enumerate(queries) for law in query.filed
        ],
    }


def _random_recall(chunks: list, units: Units, k: int) -> float:
    """Expected recall@k if k chunks were drawn blind, computed exactly."""
    total = len(chunks)
    per_law = Counter(chunk.law_number for chunk in chunks)

    hits = 0.0
    for _, laws in units:
        relevant = sum(per_law[law] for law in laws)
        misses = comb(total - relevant, k) if total - relevant >= k else 0
        hits += 1 - misses / comb(total, k)
    return hits / len(units)


def _majority_recall(units: Units) -> tuple[float, int]:
    """Recall of always answering from whichever Law appears in the most gold sets."""
    law, _ = Counter(law for _, laws in units for law in laws).most_common(1)[0]
    return sum(law in laws for _, laws in units) / len(units), law


def _recall(units: Units, retrieved: list[list[int]]) -> dict[int, float]:
    correct = dict.fromkeys(KS, 0)
    for query_index, laws in units:
        returned = retrieved[query_index]
        for k in KS:
            correct[k] += bool(set(returned[:k]) & laws)
    return {k: correct[k] / len(units) for k in KS}


def _per_law(units: Units, retrieved: list[list[int]], k: int) -> tuple[dict, dict, dict]:
    """Two different questions per Law, and they diverge a long way.

    "answered" is the headline metric restricted to one Law's traffic. "present"
    is whether that Law itself surfaced, which is what a citation needs.
    """
    total: dict[int, int] = defaultdict(int)
    answered: dict[int, int] = defaultdict(int)
    present: dict[int, int] = defaultdict(int)

    for query_index, laws in units:
        top = set(retrieved[query_index][:k])
        for law in laws:
            total[law] += 1
            answered[law] += bool(top & laws)
            present[law] += law in top

    order = sorted(total)
    return (
        dict(sorted(total.items())),
        {law: answered[law] / total[law] for law in order},
        {law: present[law] / total[law] for law in order},
    )


def evaluate(limit: int | None = None) -> dict:
    chunks = chunker.load()
    queries = gold.build(_load(FAQS))
    if limit:
        queries = queries[:limit]

    vectors = embedder.encode_queries([query.question for query in queries])
    top_k = max(KS)
    detail_k = min(KS)
    keys = _keys(queries)

    with store.connect() as connection:
        if store.count(connection) == 0:
            raise RuntimeError("index is empty, run `make index` first")

        dense = retrieve.Dense(connection)
        lexical = retrieve.Lexical(chunks)
        hybrid = retrieve.Hybrid(dense, lexical)
        retrievers = [dense, lexical, hybrid, retrieve.Reranked(hybrid)]

        retrieved = {
            retriever.name: [
                [hit.law_number for hit in retriever.search(query.question, vector, top_k)]
                for query, vector in zip(queries, vectors)
            ]
            for retriever in retrievers
        }

    results = {}
    for name, laws in retrieved.items():
        counts, answered, present = _per_law(keys["gold"], laws, detail_k)
        results[name] = {
            "recall": _recall(keys["gold"], laws),
            "recall_under_older_keys": {
                key: _recall(units, laws) for key, units in keys.items() if key != "gold"
            },
            "per_law_answered_at_detail_k": answered,
            "per_law_present_at_detail_k": present,
        }

    majority, majority_law = _majority_recall(keys["gold"])
    return {
        "model": embedder.MODEL_NAME,
        "dimensions": embedder.dimensions(),
        "chunks": len(chunks),
        "queries": len(queries),
        "cross_listed_queries": sum(len(q.crosslisted) > 1 for q in queries),
        "relabelled_queries": sum(q.gold != q.crosslisted for q in queries),
        "rrf_k": retrieve.RRF_K,
        "fusion_depth": retrieve.DEPTH,
        "headline": HEADLINE,
        "retrievers": results,
        "per_law_queries": counts,
        "baseline_random": {k: _random_recall(chunks, keys["gold"], k) for k in KS},
        "baseline_majority_law": {"law": majority_law, "recall@any": majority},
        "detail_k": detail_k,
    }


def report(result: dict) -> None:
    names = list(result["retrievers"])
    print(f"model    {result['model']} ({result['dimensions']}d)")
    print(f"corpus   {result['chunks']} chunks")
    print(
        f"queries  {result['queries']} distinct questions, "
        f"{result['cross_listed_queries']} cross-listed, "
        f"{result['relabelled_queries']} relabelled"
    )
    print(f"fusion   RRF k={result['rrf_k']} over {result['fusion_depth']} candidates each\n")

    head = "".join(f"{name:>9}" for name in names)
    print(f"{'k':>3}{head}{'random':>9}{'lift':>7}")
    for k in KS:
        row = "".join(f"{result['retrievers'][n]['recall'][k]:>8.1%} " for n in names)
        rand = result["baseline_random"][k]
        best = result["retrievers"][result["headline"]]["recall"][k]
        print(f"{k:>3}{row}{rand:>8.1%} {best / rand:>5.1f}x")

    majority = result["baseline_majority_law"]
    print(
        f"\nalways answering from Law {majority['law']} would score "
        f"{majority['recall@any']:.1%} at any k"
    )

    older = result["retrievers"][result["headline"]]["recall_under_older_keys"]
    print(f"\n{result['headline']} under the answer keys this one replaced:")
    for key, recall in older.items():
        scores = "  ".join(f"@{k} {recall[k]:.1%}" for k in KS)
        print(f"  {key:>12}  {scores}")

    k = result["detail_k"]
    print(f"\nquestions answered@{k}, by Law in the gold set:")
    print(f"  {'Law':>6} {'n':>4}{head}")
    for law, n in result["per_law_queries"].items():
        row = "".join(
            f"{result['retrievers'][name]['per_law_answered_at_detail_k'][law]:>8.1%} "
            for name in names
        )
        print(f"  Law {law:>2} {n:>4}{row}")


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

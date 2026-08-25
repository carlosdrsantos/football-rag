"""Cross-encoder reranking of the candidates hybrid retrieval already found.

The bi-encoder and BM25 both score a question against a chunk without ever
looking at the two together: the chunk is reduced to 384 numbers, or to a bag of
stemmed terms, before the question arrives. A cross-encoder reads the pair as one
sequence, so it can tell that "when can a coach be sent off" wants TEAM OFFICIALS
rather than PLAYERS. It costs a full forward pass per candidate, which is why it
reranks the top few instead of searching all 139.

Depth 10 because hybrid recall@10 is 98.8%: the answer is nearly always inside
what the cross-encoder gets to see, and reranking deeper measurably buys nothing.
Going to 25 leaves recall@1 unchanged and costs 728 ms a query instead of 291.
"""

import os
from functools import cache

from sentence_transformers import CrossEncoder

MODEL_NAME = os.environ.get("LOTG_RERANKER", "BAAI/bge-reranker-base")


@cache
def model(name: str = MODEL_NAME) -> CrossEncoder:
    return CrossEncoder(name, max_length=512)


def score(question: str, passages: list[str], name: str = MODEL_NAME) -> list[float]:
    """Relevance of each passage to the question. Higher is better."""
    if not passages:
        return []
    pairs = [(question, passage) for passage in passages]
    return [float(value) for value in model(name).predict(pairs, show_progress_bar=False)]

"""The three retrievers, behind one interface so the eval can score them together.

Dense and BM25 rank on scales that have nothing to say to each other, so they are
fused on rank rather than score. Reciprocal rank fusion gives a document
1 / (RRF_K + rank) from each list it appears in and sums. RRF_K = 60 is the
value from the paper that introduced it, and it is the only knob: raising it
flattens the weight given to the top of each list.

Each retriever is asked for DEPTH candidates before fusion, because a chunk
ranked 30th by one and 2nd by the other is exactly what fusion is for.
"""

from collections import defaultdict
from dataclasses import dataclass, replace

import psycopg

from lotg.ingest.chunk import Chunk, embed_text
from lotg.retrieval import bm25, store
from lotg.retrieval.store import Hit

RRF_K = 60
DEPTH = 50


@dataclass
class Dense:
    connection: psycopg.Connection
    name = "dense"

    def search(self, question: str, vector: list[float], limit: int) -> list[Hit]:
        return store.search(self.connection, vector, limit)


class Lexical:
    name = "lexical"

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.index = bm25.build([embed_text(chunk) for chunk in chunks])

    def search(self, question: str, vector: list[float], limit: int) -> list[Hit]:
        hits = []
        for index, score in self.index.search(question, limit):
            chunk = self.chunks[index]
            hits.append(
                Hit(chunk.id, chunk.law_number, chunk.breadcrumb, chunk.body, chunk.url, score)
            )
        return hits


@dataclass
class Hybrid:
    dense: Dense
    lexical: Lexical
    depth: int = DEPTH
    rrf_k: int = RRF_K
    name = "hybrid"

    def search(self, question: str, vector: list[float], limit: int) -> list[Hit]:
        rankings = [
            self.dense.search(question, vector, self.depth),
            self.lexical.search(question, vector, self.depth),
        ]
        found = {hit.id: hit for hits in rankings for hit in hits}
        fused = fuse([[hit.id for hit in hits] for hits in rankings], self.rrf_k)
        return [replace(found[chunk_id], score=score) for chunk_id, score in fused[:limit]]


def fuse(rankings: list[list[str]], rrf_k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion. Ties break on id so a run is reproducible."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))

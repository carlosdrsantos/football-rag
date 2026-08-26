"""The HTTP service.

Both models and the BM25 index load once at startup, not per request: the
cross-encoder alone is 278M parameters and loading it per call would cost more
than everything else combined. Postgres connections come from a pool, because a
connection held open for the life of the process dies quietly and takes the
service with it.

/search is the retrieval stack on its own and needs no API key. /ask puts a model
on top of it and needs one.
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lotg.ingest import chunk as chunker
from lotg.retrieval import embedder, reranker, retrieve, store
from lotg.retrieval.store import Hit
from lotg.service import answer as answering

TOP_K = int(os.environ.get("LOTG_TOP_K", "5"))


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    k: int = Field(default=TOP_K, ge=1, le=20)


class Clause(BaseModel):
    id: str
    law: int
    breadcrumb: str
    text: str
    url: str
    score: float

    @classmethod
    def of(cls, hit: Hit) -> "Clause":
        return cls(
            id=hit.id,
            law=hit.law_number,
            breadcrumb=hit.breadcrumb,
            text=hit.body,
            url=hit.url,
            score=hit.score,
        )


class SearchResponse(BaseModel):
    question: str
    clauses: list[Clause]
    took_ms: int


class AskResponse(BaseModel):
    question: str
    answer: str
    sufficient: bool
    citations: list[Clause]
    considered: list[Clause]
    took_ms: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    chunks = chunker.load()
    app.state.pool = store.pool()
    app.state.lexical = retrieve.Lexical(chunks)
    app.state.chunks = len(chunks)
    embedder.dimensions()  # pull the encoder into memory before the first request
    reranker.model()
    yield
    app.state.pool.close()


app = FastAPI(
    title="Laws of the Game RAG",
    description="Question answering over the IFAB Laws of the Game, with citations.",
    lifespan=lifespan,
)


def _retrieve(question: str, k: int) -> list[Hit]:
    vector = embedder.encode_queries([question])[0]
    with app.state.pool.connection() as connection:
        dense = retrieve.Dense(connection)
        stack = retrieve.Reranked(retrieve.Hybrid(dense, app.state.lexical))
        return stack.search(question, vector, k)


@app.get("/health")
def health() -> dict:
    with app.state.pool.connection() as connection:
        indexed = store.count(connection)
    return {
        "status": "ok" if indexed else "index is empty",
        "indexed_chunks": indexed,
        "embedder": embedder.MODEL_NAME,
        "reranker": reranker.MODEL_NAME,
        "generator": answering.MODEL if os.environ.get("ANTHROPIC_API_KEY") else None,
    }


@app.post("/search", response_model=SearchResponse)
def search(request: Question) -> SearchResponse:
    started = time.perf_counter()
    hits = _retrieve(request.question, request.k)
    return SearchResponse(
        question=request.question,
        clauses=[Clause.of(hit) for hit in hits],
        took_ms=round((time.perf_counter() - started) * 1000),
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: Question) -> AskResponse:
    started = time.perf_counter()
    hits = _retrieve(request.question, request.k)
    try:
        ruling, cited = answering.answer(request.question, hits)
    except RuntimeError as missing_key:
        raise HTTPException(status_code=503, detail=str(missing_key)) from missing_key

    return AskResponse(
        question=request.question,
        answer=ruling.answer,
        sufficient=ruling.sufficient,
        citations=[Clause.of(hit) for hit in cited],
        considered=[Clause.of(hit) for hit in hits],
        took_ms=round((time.perf_counter() - started) * 1000),
    )

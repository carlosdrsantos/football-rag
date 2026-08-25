"""pgvector-backed chunk store."""

import os
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector

from lotg.ingest.chunk import Chunk

DSN = os.environ.get("LOTG_DSN", "postgresql://lotg:lotg@localhost:5433/lotg")

SCHEMA = """
DROP TABLE IF EXISTS chunks;
CREATE TABLE chunks (
    id           TEXT PRIMARY KEY,
    section_id   TEXT NOT NULL,
    law_number   INT  NOT NULL,
    law_title    TEXT NOT NULL,
    clause       TEXT,
    clause_title TEXT NOT NULL,
    group_label  TEXT,
    subsection   TEXT,
    breadcrumb   TEXT NOT NULL,
    body         TEXT NOT NULL,
    url          TEXT NOT NULL,
    embedding    vector({dims}) NOT NULL
);

CREATE INDEX chunks_embedding_idx ON chunks
    USING hnsw (embedding vector_cosine_ops);
"""


@dataclass
class Hit:
    id: str
    law_number: int
    breadcrumb: str
    body: str
    url: str
    score: float


def connect() -> psycopg.Connection:
    connection = psycopg.connect(DSN)
    # register_vector looks the type up by OID, so the extension has to exist first.
    connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    connection.commit()
    register_vector(connection)
    return connection


def create_schema(connection: psycopg.Connection, dims: int) -> None:
    connection.execute(SCHEMA.format(dims=dims))
    connection.commit()


def insert(
    connection: psycopg.Connection, chunks: list[Chunk], vectors: list[list[float]]
) -> None:
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO chunks (id, section_id, law_number, law_title, clause,
                                clause_title, group_label, subsection, breadcrumb,
                                body, url, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    chunk.id,
                    chunk.section_id,
                    chunk.law_number,
                    chunk.law_title,
                    chunk.clause,
                    chunk.clause_title,
                    chunk.group,  # stored as group_label; "group" is reserved in SQL
                    chunk.subsection,
                    chunk.breadcrumb,
                    chunk.body,
                    chunk.url,
                    vector,
                )
                for chunk, vector in zip(chunks, vectors)
            ],
        )
    connection.commit()


def search(connection: psycopg.Connection, vector: list[float], limit: int) -> list[Hit]:
    # <=> is cosine distance, so similarity is 1 - distance.
    rows = connection.execute(
        """
        SELECT id, law_number, breadcrumb, body, url, 1 - (embedding <=> %s::vector)
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (vector, vector, limit),
    ).fetchall()
    return [Hit(*row) for row in rows]


def count(connection: psycopg.Connection) -> int:
    return connection.execute("SELECT count(*) FROM chunks").fetchone()[0]

"""Embed chunks.jsonl and load it into pgvector."""

import json
from pathlib import Path

from lotg.ingest.chunk import Chunk, embed_text
from lotg.retrieval import embedder, store

CHUNKS = Path("data/processed/chunks.jsonl")


def main() -> None:
    if not CHUNKS.exists():
        raise FileNotFoundError(f"{CHUNKS} missing, run `make chunk` first")

    chunks = [Chunk(**json.loads(line)) for line in CHUNKS.open(encoding="utf-8")]
    print(f"embedding {len(chunks)} chunks with {embedder.MODEL_NAME}")
    vectors = embedder.encode_documents([embed_text(c) for c in chunks])

    with store.connect() as connection:
        store.create_schema(connection, embedder.dimensions())
        store.insert(connection, chunks, vectors)
        print(f"indexed {store.count(connection)} chunks, {embedder.dimensions()} dimensions")


if __name__ == "__main__":
    main()

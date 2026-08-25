"""Embed chunks.jsonl and load it into pgvector."""

from lotg.ingest import chunk as chunker
from lotg.ingest.chunk import embed_text
from lotg.retrieval import embedder, store


def main() -> None:
    chunks = chunker.load()
    print(f"embedding {len(chunks)} chunks with {embedder.MODEL_NAME}")
    vectors = embedder.encode_documents([embed_text(c) for c in chunks])

    with store.connect() as connection:
        store.create_schema(connection, embedder.dimensions())
        store.insert(connection, chunks, vectors)
        print(f"indexed {store.count(connection)} chunks, {embedder.dimensions()} dimensions")


if __name__ == "__main__":
    main()

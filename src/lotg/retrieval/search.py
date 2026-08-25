"""Query the index from the command line."""

import argparse

from lotg.ingest import chunk as chunker
from lotg.retrieval import embedder, retrieve, store


def build(connection, name: str):
    dense = retrieve.Dense(connection)
    if name == "dense":
        return dense
    lexical = retrieve.Lexical(chunker.load())
    return lexical if name == "lexical" else retrieve.Hybrid(dense, lexical)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the Laws of the Game.")
    parser.add_argument("query", nargs="+")
    parser.add_argument("-k", type=int, default=5, help="how many chunks to return")
    parser.add_argument(
        "-r", "--retriever", default="hybrid", choices=("hybrid", "dense", "lexical")
    )
    args = parser.parse_args()

    query = " ".join(args.query)
    vector = embedder.encode_queries([query])[0]

    with store.connect() as connection:
        hits = build(connection, args.retriever).search(query, vector, args.k)

    print(f"query: {query}  ({args.retriever})\n")
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}. [{hit.score:.4f}] {hit.breadcrumb}")
        print(f"   {hit.body[:200].strip()}...\n")


if __name__ == "__main__":
    main()

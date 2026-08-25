"""Query the index from the command line."""

import argparse

from lotg.retrieval import embedder, store


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the Laws of the Game.")
    parser.add_argument("query", nargs="+")
    parser.add_argument("-k", type=int, default=5, help="how many chunks to return")
    args = parser.parse_args()

    query = " ".join(args.query)
    vector = embedder.encode_queries([query])[0]

    with store.connect() as connection:
        hits = store.search(connection, vector, args.k)

    print(f"query: {query}\n")
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}. [{hit.score:.3f}] {hit.breadcrumb}")
        print(f"   {hit.body[:200].strip()}...\n")


if __name__ == "__main__":
    main()

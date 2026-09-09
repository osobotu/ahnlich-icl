import argparse
import asyncio
import os

from ahnlich_icl.ahnlich_mcp import similarity_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store_name = os.getenv("AHNLICH_STORE_NAME", "spider_examples")

    results = asyncio.run(
        similarity_search(store_name, args.question, args.k)
    )

    print(f"Query:\n{args.question}\n")

    for rank, match in enumerate(results, start=1):
        example = match.example

        print(f"{rank}. similarity={match.similarity:.4f}")
        print(f"   ID: {example.example_id}")
        print(f"   Database: {example.db_id}")
        print(f"   Question: {example.question}")
        print(f"   SQL: {example.sql}")
        print()


if __name__ == "__main__":
    main()
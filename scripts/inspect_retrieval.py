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

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print(f"{rank}. similarity={result['similarity']:.4f}")
        print(f"   ID: {metadata['example_id']}")
        print(f"   Database: {metadata['db_id']}")
        print(f"   Question: {result['content']}")
        print(f"   SQL: {metadata['sql']}")
        print()


if __name__ == "__main__":
    main()
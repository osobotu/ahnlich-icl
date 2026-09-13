import argparse
import asyncio
import os
import sqlite3
from pathlib import Path

from openai import OpenAIError

from ahnlich_icl.agent import AgentAnswer, answer_question
from ahnlich_icl.spider import database_path, load_schema


EXIT_COMMANDS = {"exit", "quit"}


def format_answer(answer: AgentAnswer) -> str:
    example_count = len(answer.retrieved_examples)
    if example_count:
        noun = "example" if example_count == 1 else "examples"
        retrieval = f"Ahnlich MCP ({example_count} {noun})"
    else:
        retrieval = "not used"

    lines = [f"Retrieval: {retrieval}"]

    for match in answer.retrieved_examples:
        lines.append(
            f"  {match.similarity:.4f} "
            f"{match.example.question}"
        )

    lines.extend(
        [
            "",
            "SQL:",
            answer.sql,
            "",
            "Result:",
        ]
    )

    if answer.columns:
        lines.append(" | ".join(answer.columns))

    if answer.rows:
        lines.extend(
            " | ".join(_format_value(value) for value in row)
            for row in answer.rows
        )
    else:
        lines.append("(no rows)")

    if answer.truncated:
        lines.append("(only the first 100 rows are shown)")

    return "\n".join(lines)


async def run_chat(
    spider_dir: Path,
    db_id: str,
    store_name: str,
) -> None:
    schema = load_schema(spider_dir, db_id)
    sqlite_path = database_path(spider_dir, db_id)

    print(f"Database: {db_id}")
    print("Ask a question, or type 'quit' to stop.")

    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if question.lower() in EXIT_COMMANDS:
            return

        if not question:
            continue

        print(f"\nQuestion: {question}")

        try:
            answer = await answer_question(
                question=question,
                schema=schema,
                database_path=sqlite_path,
                store_name=store_name,
            )
        except (OpenAIError, RuntimeError, sqlite3.Error) as error:
            print(f"Error: {error}")
            continue

        print(format_answer(answer))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask natural-language questions about a Spider database."
    )
    parser.add_argument(
        "db_id",
        help="Spider database ID, for example concert_singer",
    )
    parser.add_argument(
        "--spider-dir",
        type=Path,
        default=Path(os.getenv("SPIDER_DIR", "data/spider")),
    )
    parser.add_argument(
        "--store-name",
        default=os.getenv(
            "AHNLICH_STORE_NAME",
            "spider_examples",
        ),
    )
    args = parser.parse_args()

    asyncio.run(
        run_chat(
            spider_dir=args.spider_dir,
            db_id=args.db_id,
            store_name=args.store_name,
        )
    )


def _format_value(value: object) -> str:
    return "NULL" if value is None else str(value)


if __name__ == "__main__":
    main()
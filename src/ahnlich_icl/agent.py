import sqlite3
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from ahnlich_icl.ahnlich_mcp import SearchMatch, similarity_search
from ahnlich_icl.text2sql import generate_sql


MAX_RESULT_ROWS = 100
QUERY_TIMEOUT_SECONDS = 5.0
PROGRESS_HANDLER_STEPS = 10_000
MIN_EXAMPLE_SIMILARITY = 0.5


@dataclass(frozen=True)
class AgentAnswer:
    sql: str
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    retrieved_examples: tuple[SearchMatch, ...]
    truncated: bool


async def answer_question(
    question: str,
    schema: str,
    database_path: Path,
    store_name: str,
) -> AgentAnswer:
    """Retrieve examples, falling back to zero-shot when none qualify."""
    matches = await similarity_search(
        store_name,
        question,
        k=5,
    )

    retrieved_examples = tuple(
        match
        for match in matches
        if match.similarity >= MIN_EXAMPLE_SIMILARITY
    )

    sql = generate_sql(
        question,
        schema,
        demonstrations=[
            match.example
            for match in retrieved_examples
        ],
    )

    columns, rows, truncated = _execute_read_only(
        database_path,
        sql,
    )

    return AgentAnswer(
        sql=sql,
        columns=columns,
        rows=rows,
        retrieved_examples=retrieved_examples,
        truncated=truncated,
    )

def _execute_read_only(
    database_path: Path,
    sql: str,
) -> tuple[
    tuple[str, ...],
    tuple[tuple[object, ...], ...],
    bool,
]:
    deadline = monotonic() + QUERY_TIMEOUT_SECONDS
    database_uri = database_path.resolve().as_uri() + "?mode=ro"

    with sqlite3.connect(database_uri, uri=True) as connection:
        connection.set_progress_handler(
            lambda: monotonic() >= deadline,
            PROGRESS_HANDLER_STEPS,
        )
        cursor = connection.execute(sql)
        columns = tuple(
            description[0]
            for description in cursor.description or ()
        )
        rows = tuple(cursor.fetchmany(MAX_RESULT_ROWS + 1))

    return columns, rows[:MAX_RESULT_ROWS], len(rows) > MAX_RESULT_ROWS

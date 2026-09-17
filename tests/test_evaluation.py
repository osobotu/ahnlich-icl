import sqlite3
from pathlib import Path

from ahnlich_icl.evaluation import _prediction_error


def create_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE employees (name TEXT, salary INTEGER)"
        )
        connection.executemany(
            "INSERT INTO employees VALUES (?, ?)",
            [("Ada", 70_000), ("Linus", 60_000), ("Grace", 50_000)],
        )


def test_invalid_generated_sql_returns_an_error(tmp_path: Path) -> None:
    database_path = tmp_path / "company.sqlite"
    create_database(database_path)

    error = _prediction_error(
        database_path,
        "SELECT missing_column FROM employees",
    )

    assert "missing_column" in error


def test_generated_sql_cannot_modify_the_database(tmp_path: Path) -> None:
    database_path = tmp_path / "company.sqlite"
    create_database(database_path)

    error = _prediction_error(
        database_path,
        "DELETE FROM employees",
    )

    with sqlite3.connect(database_path) as connection:
        row_count = connection.execute("SELECT count(*) FROM employees").fetchone()

    assert error is not None
    assert row_count == (3,)


def test_long_generated_sql_is_interrupted(tmp_path: Path) -> None:
    database_path = tmp_path / "company.sqlite"
    create_database(database_path)
    endless_query = """
        WITH RECURSIVE numbers(value) AS (
            SELECT 1
            UNION ALL
            SELECT value + 1 FROM numbers
        )
        SELECT max(value) FROM numbers
    """

    error = _prediction_error(
        database_path,
        endless_query,
        timeout_seconds=0.001,
    )

    assert "interrupted" in error.lower()

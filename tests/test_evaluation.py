import sqlite3
from pathlib import Path

from ahnlich_icl.evaluation import evaluate_execution


def create_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE employees (name TEXT, salary INTEGER)"
        )
        connection.executemany(
            "INSERT INTO employees VALUES (?, ?)",
            [("Ada", 70_000), ("Linus", 60_000), ("Grace", 50_000)],
        )


def test_equivalent_result_rows_match_regardless_of_order(tmp_path: Path) -> None:
    database_path = tmp_path / "company.sqlite"
    create_database(database_path)

    result = evaluate_execution(
        database_path,
        "SELECT name FROM employees ORDER BY name DESC",
        "SELECT name FROM employees",
    )

    assert result.execution_match is True
    assert result.error is None


def test_different_result_rows_do_not_match(tmp_path: Path) -> None:
    database_path = tmp_path / "company.sqlite"
    create_database(database_path)

    result = evaluate_execution(
        database_path,
        "SELECT name FROM employees WHERE salary > 60000",
        "SELECT name FROM employees WHERE salary >= 60000",
    )

    assert result.execution_match is False
    assert result.error is None


def test_invalid_generated_sql_is_a_failed_result(tmp_path: Path) -> None:
    database_path = tmp_path / "company.sqlite"
    create_database(database_path)

    result = evaluate_execution(
        database_path,
        "SELECT missing_column FROM employees",
        "SELECT name FROM employees",
    )

    assert result.execution_match is False
    assert "missing_column" in result.error


def test_generated_sql_cannot_modify_the_database(tmp_path: Path) -> None:
    database_path = tmp_path / "company.sqlite"
    create_database(database_path)

    result = evaluate_execution(
        database_path,
        "DELETE FROM employees",
        "SELECT name FROM employees",
    )

    with sqlite3.connect(database_path) as connection:
        row_count = connection.execute("SELECT count(*) FROM employees").fetchone()

    assert result.execution_match is False
    assert result.error is not None
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

    result = evaluate_execution(
        database_path,
        endless_query,
        "SELECT name FROM employees",
        timeout_seconds=0.001,
    )

    assert result.execution_match is False
    assert "interrupted" in result.error.lower()

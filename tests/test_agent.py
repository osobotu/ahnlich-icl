import asyncio
import sqlite3
from pathlib import Path

import pytest

import ahnlich_icl.agent as agent
from ahnlich_icl.ahnlich_mcp import SearchMatch
from ahnlich_icl.spider import Example
from ahnlich_icl.text2sql import SqlGeneration


def create_database(tmp_path: Path) -> Path:
    database_path = tmp_path / "company.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE employees (name TEXT, salary INTEGER)"
        )
        connection.executemany(
            "INSERT INTO employees VALUES (?, ?)",
            [("Ada", 90_000), ("Linus", 70_000)],
        )
    return database_path


def retrieved_example() -> SearchMatch:
    return SearchMatch(
        example=Example(
            example_id="train:00001",
            question="Which workers earn more than 50000?",
            sql="SELECT name FROM workers WHERE salary > 50000",
            db_id="workplace",
        ),
        similarity=0.88,
    )


def test_always_retrieves_examples_before_generating_sql(
    tmp_path,
    monkeypatch,
):
    database_path = create_database(tmp_path)
    match = retrieved_example()
    rejected_match = SearchMatch(
        example=match.example,
        similarity=0.49,
    )
    calls = []

    async def search(store_name: str, question: str, k: int):
        calls.append("search")
        assert (store_name, question, k) == (
            "spider_examples",
            "Which employees earn more than 80000?",
            5,
        )
        return [match, rejected_match]

    def generate(question, schema, demonstrations):
        calls.append("generate")
        assert demonstrations == [match.example]
        return SqlGeneration(
            raw_output="SELECT name FROM employees WHERE salary > 80000",
            sql="SELECT name FROM employees WHERE salary > 80000",
        )

    monkeypatch.setattr(agent, "similarity_search", search)
    monkeypatch.setattr(agent, "generate_sql", generate)

    answer = asyncio.run(
        agent.answer_question(
            question="Which employees earn more than 80000?",
            schema="CREATE TABLE employees (name TEXT, salary INTEGER);",
            database_path=database_path,
            store_name="spider_examples",
        )
    )

    assert calls == ["search", "generate"]
    assert answer.sql == "SELECT name FROM employees WHERE salary > 80000"
    assert answer.columns == ("name",)
    assert answer.rows == (("Ada",),)
    assert answer.retrieved_examples == (match,)


def test_generates_without_examples_when_none_meet_the_threshold(
    tmp_path,
    monkeypatch,
):
    database_path = create_database(tmp_path)
    match = SearchMatch(
        example=retrieved_example().example,
        similarity=0.49,
    )

    async def search(store_name: str, question: str, k: int):
        return [match]

    def generate(question, schema, demonstrations):
        assert demonstrations == []
        return SqlGeneration(
            raw_output="SELECT name FROM employees",
            sql="SELECT name FROM employees",
        )

    monkeypatch.setattr(agent, "similarity_search", search)
    monkeypatch.setattr(agent, "generate_sql", generate)

    answer = asyncio.run(
        agent.answer_question(
            question="Which employees earn more than 80000?",
            schema="CREATE TABLE employees (name TEXT, salary INTEGER);",
            database_path=database_path,
            store_name="spider_examples",
        )
    )

    assert answer.rows == (("Ada",), ("Linus",))
    assert answer.retrieved_examples == ()


def test_generated_sql_cannot_modify_the_database(tmp_path, monkeypatch):
    database_path = create_database(tmp_path)

    async def search(store_name: str, question: str, k: int):
        return [retrieved_example()]

    def generate(question, schema, demonstrations):
        return SqlGeneration(
            raw_output="DELETE FROM employees",
            sql="DELETE FROM employees",
        )

    monkeypatch.setattr(agent, "similarity_search", search)
    monkeypatch.setattr(agent, "generate_sql", generate)

    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        asyncio.run(
            agent.answer_question(
                question="Delete every employee.",
                schema="CREATE TABLE employees (name TEXT, salary INTEGER);",
                database_path=database_path,
                store_name="spider_examples",
            )
        )

import json
import sqlite3

import pytest

from ahnlich_icl.spider import load_examples, load_schema


def test_loads_examples_with_stable_ids(tmp_path):
    rows = [
        {
            "question": "How many employees are there?",
            "query": "SELECT COUNT(*) FROM employees",
            "db_id": "company",
        },
        {
            "question": "List every employee name.",
            "query": "SELECT name FROM employees",
            "db_id": "company",
        },
    ]
    (tmp_path / "train_spider.json").write_text(
        json.dumps(rows),
        encoding="utf-8",
    )

    examples = load_examples(tmp_path, "train")

    assert examples[0].example_id == "train:00001"
    assert examples[0].question == "How many employees are there?"
    assert examples[0].sql == "SELECT COUNT(*) FROM employees"
    assert examples[0].db_id == "company"
    assert examples[0].difficulty is None
    assert examples[1].example_id == "train:00002"


def test_reports_a_missing_split(tmp_path):
    with pytest.raises(FileNotFoundError, match="dev.json"):
        load_examples(tmp_path, "dev")


def test_loads_schema_from_sqlite(tmp_path):
    database_dir = tmp_path / "database" / "company"
    database_dir.mkdir(parents=True)
    database = database_dir / "company.sqlite"

    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE departments (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT,
                department_id INTEGER,
                FOREIGN KEY (department_id) REFERENCES departments(id)
            )
            """
        )

    schema = load_schema(tmp_path, "company")

    assert "CREATE TABLE departments" in schema
    assert "CREATE TABLE employees" in schema
    assert "FOREIGN KEY (department_id)" in schema
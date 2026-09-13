import asyncio
import sqlite3
from pathlib import Path

import ahnlich_icl.agent_cli as agent_cli
from ahnlich_icl.agent import AgentAnswer
from ahnlich_icl.ahnlich_mcp import SearchMatch
from ahnlich_icl.spider import Example


def test_formats_sql_rows_and_retrieval_details():
    retrieved = SearchMatch(
        example=Example(
            example_id="train:00001",
            question="Which workers earn more than 50000?",
            sql="SELECT name FROM workers WHERE salary > 50000",
            db_id="workplace",
        ),
        similarity=0.88,
    )
    answer = AgentAnswer(
        sql="SELECT name, salary FROM employees",
        columns=("name", "salary"),
        rows=(("Ada", 90_000), ("Linus", None)),
        retrieved_examples=(retrieved,),
        truncated=False,
    )

    output = agent_cli.format_answer(answer)

    assert "Retrieval: Ahnlich MCP (1 example)" in output
    assert "0.8800 Which workers earn more than 50000?" in output
    assert "SELECT name, salary FROM employees" in output
    assert "name | salary" in output
    assert "Ada | 90000" in output
    assert "Linus | NULL" in output


def test_reports_no_examples_used_after_retrieval():
    answer = AgentAnswer(
        sql="SELECT COUNT(*) FROM employees",
        columns=("COUNT(*)",),
        rows=((2,),),
        retrieved_examples=(),
        truncated=False,
    )

    output = agent_cli.format_answer(answer)

    assert "Retrieval: not used" in output


def test_chat_accepts_multiple_independent_questions(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    database_dir = tmp_path / "database" / "company"
    database_dir.mkdir(parents=True)
    database_path = database_dir / "company.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE employees (name TEXT)")

    questions = iter(
        [
            "How many employees are there?",
            "List their names.",
            "quit",
        ]
    )
    calls = []

    async def answer_question(
        question,
        schema,
        database_path,
        store_name,
    ):
        calls.append(
            (question, schema, database_path, store_name)
        )
        return AgentAnswer(
            sql=f"SELECT '{question}'",
            columns=("answer",),
            rows=((question,),),
            retrieved_examples=(),
            truncated=False,
        )

    monkeypatch.setattr("builtins.input", lambda prompt: next(questions))
    monkeypatch.setattr(agent_cli, "answer_question", answer_question)

    asyncio.run(
        agent_cli.run_chat(
            spider_dir=tmp_path,
            db_id="company",
            store_name="spider_examples",
        )
    )

    assert [call[0] for call in calls] == [
        "How many employees are there?",
        "List their names.",
    ]
    assert all("CREATE TABLE employees" in call[1] for call in calls)
    assert all(call[2] == database_path for call in calls)
    assert all(call[3] == "spider_examples" for call in calls)

    output = capsys.readouterr().out
    assert "Database: company" in output
    assert "How many employees are there?" in output
    assert "List their names." in output

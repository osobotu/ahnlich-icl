import json

from ahnlich_icl.experiment import (
    append_jsonl,
    build_generation_record,
    summarize_records,
)
from ahnlich_icl.evaluation import SpiderEvaluation
from ahnlich_icl.retrieval import SelectedExample
from ahnlich_icl.spider import Example


def test_builds_generation_record():
    target = Example(
        example_id="dev:00001",
        question="How many singers are there?",
        sql="SELECT COUNT(*) FROM singer",
        db_id="concert_singer",
    )
    demonstration = Example(
        example_id="train:00010",
        question="How many employees are there?",
        sql="SELECT COUNT(*) FROM employee",
        db_id="company",
    )
    selected = [
        SelectedExample(
            example=demonstration,
            similarity=0.82,
        )
    ]
    evaluation = SpiderEvaluation(
        difficulty="easy",
        test_suite_correct=True,
        exact_match=True,
    )

    record = build_generation_record(
        target,
        method="similar",
        selected_examples=selected,
        generated_sql="SELECT COUNT(*) FROM singer;",
        evaluation=evaluation,
    )

    assert record == {
        "example_id": "dev:00001",
        "db_id": "concert_singer",
        "difficulty": "easy",
        "question": "How many singers are there?",
        "gold_sql": "SELECT COUNT(*) FROM singer",
        "method": "similar",
        "selected_examples": [
            {
                "example_id": "train:00010",
                "question": "How many employees are there?",
                "sql": "SELECT COUNT(*) FROM employee",
                "db_id": "company",
                "similarity": 0.82,
            }
        ],
        "generated_sql": "SELECT COUNT(*) FROM singer;",
        "test_suite_correct": True,
        "exact_match": True,
        "evaluation_error": None,
    }


def test_records_structured_evaluation_failure():
    target = Example(
        example_id="dev:00002",
        question="Return a missing column.",
        sql="SELECT name FROM singer",
        db_id="concert_singer",
    )
    evaluation = SpiderEvaluation(
        difficulty="medium",
        test_suite_correct=False,
        exact_match=False,
        error="no such column: missing",
    )

    record = build_generation_record(
        target,
        method="zero",
        selected_examples=[],
        generated_sql="SELECT missing FROM singer",
        evaluation=evaluation,
    )

    assert record["difficulty"] == "medium"
    assert record["test_suite_correct"] is False
    assert record["exact_match"] is False
    assert record["evaluation_error"] == "no such column: missing"


def test_appends_one_json_object_per_line(tmp_path):
    output_path = tmp_path / "results" / "pilot.jsonl"
    first = {"example_id": "dev:00001"}
    second = {"example_id": "dev:00002"}

    append_jsonl(output_path, first)
    append_jsonl(output_path, second)

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert [json.loads(line) for line in lines] == [first, second]


def test_summarizes_each_method_overall_and_by_difficulty():
    records = [
        result_record("zero", "easy", test_suite=True, exact=True),
        result_record("zero", "hard", test_suite=False, exact=False),
        result_record("similar", "easy", test_suite=False, exact=False),
        result_record(
            "similar",
            None,
            test_suite=False,
            exact=False,
            error="timeout",
        ),
    ]

    summary = summarize_records(records)

    assert summary["zero"] == {
        "count": 2,
        "test_suite_accuracy": 0.5,
        "exact_match_accuracy": 0.5,
        "evaluation_errors": 0,
        "by_difficulty": {
            "easy": {
                "count": 1,
                "test_suite_accuracy": 1.0,
                "exact_match_accuracy": 1.0,
                "evaluation_errors": 0,
            },
            "hard": {
                "count": 1,
                "test_suite_accuracy": 0.0,
                "exact_match_accuracy": 0.0,
                "evaluation_errors": 0,
            },
        },
    }
    assert summary["similar"]["test_suite_accuracy"] == 0.0
    assert summary["similar"]["evaluation_errors"] == 1
    assert (
        summary["similar"]["by_difficulty"]["unknown"]["evaluation_errors"]
        == 1
    )


def test_empty_records_produce_an_empty_summary():
    assert summarize_records([]) == {}


def result_record(
    method: str,
    difficulty: str | None,
    *,
    test_suite: bool,
    exact: bool,
    error: str | None = None,
) -> dict:
    return {
        "method": method,
        "difficulty": difficulty,
        "test_suite_correct": test_suite,
        "exact_match": exact,
        "evaluation_error": error,
    }

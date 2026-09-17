import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ahnlich_icl.evaluation import SpiderEvaluation
from ahnlich_icl.retrieval import (
    SelectedExample,
    SelectionMethod,
)
from ahnlich_icl.spider import Example


JsonRecord = dict[str, Any]


def build_generation_record(
    target: Example,
    *,
    method: SelectionMethod,
    selected_examples: Sequence[SelectedExample],
    generated_sql: str,
    evaluation: SpiderEvaluation,
) -> JsonRecord:
    return {
        "example_id": target.example_id,
        "db_id": target.db_id,
        "difficulty": evaluation.difficulty,
        "question": target.question,
        "gold_sql": target.sql,
        "method": method,
        "selected_examples": [
            _selected_example_record(selected)
            for selected in selected_examples
        ],
        "generated_sql": generated_sql,
        "test_suite_correct": evaluation.test_suite_correct,
        "exact_match": evaluation.exact_match,
        "evaluation_error": evaluation.error,
    }


def append_jsonl(
    output_path: Path,
    record: JsonRecord,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def summarize_records(
    records: Sequence[JsonRecord],
) -> JsonRecord:
    records_by_method: dict[str, list[JsonRecord]] = {}

    for record in records:
        method = record["method"]
        records_by_method.setdefault(method, []).append(record)

    return {
        method: {
            **_metric_summary(method_records),
            "by_difficulty": _difficulty_summaries(method_records),
        }
        for method, method_records in sorted(records_by_method.items())
    }


def _difficulty_summaries(
    records: Sequence[JsonRecord],
) -> JsonRecord:
    records_by_difficulty: dict[str, list[JsonRecord]] = {}

    for record in records:
        difficulty = record["difficulty"] or "unknown"
        records_by_difficulty.setdefault(difficulty, []).append(record)

    return {
        difficulty: _metric_summary(difficulty_records)
        for difficulty, difficulty_records
        in sorted(records_by_difficulty.items())
    }


def _metric_summary(
    records: Sequence[JsonRecord],
) -> JsonRecord:
    count = len(records)

    return {
        "count": count,
        "test_suite_accuracy": sum(
            record["test_suite_correct"] for record in records
        ) / count,
        "exact_match_accuracy": sum(
            record["exact_match"] for record in records
        ) / count,
        "evaluation_errors": sum(
            record["evaluation_error"] is not None
            for record in records
        ),
    }


def _selected_example_record(
    selected: SelectedExample,
) -> JsonRecord:
    example = selected.example

    return {
        "example_id": example.example_id,
        "question": example.question,
        "sql": example.sql,
        "db_id": example.db_id,
        "similarity": selected.similarity,
    }

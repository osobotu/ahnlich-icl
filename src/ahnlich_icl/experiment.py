import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ahnlich_icl.retrieval import (
    SelectedExample,
    SelectionMethod,
)
from ahnlich_icl.spider import Example
from ahnlich_icl.text2sql import SqlGeneration


JsonRecord = dict[str, Any]


def build_generation_record(
    target: Example,
    *,
    method: SelectionMethod,
    selected_examples: Sequence[SelectedExample],
    generation: SqlGeneration,
) -> JsonRecord:
    return {
        "example_id": target.example_id,
        "db_id": target.db_id,
        "difficulty": target.difficulty,
        "question": target.question,
        "gold_sql": target.sql,
        "method": method,
        "selected_examples": [
            _selected_example_record(selected)
            for selected in selected_examples
        ],
        "raw_output": generation.raw_output,
        "generated_sql": generation.sql,
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


def _selected_example_record(
    selected: SelectedExample,
) -> JsonRecord:
    example = selected.example

    return {
        "example_id": example.example_id,
        "question": example.question,
        "sql": example.sql,
        "db_id": example.db_id,
        "difficulty": example.difficulty,
        "source": selected.source,
        "similarity": selected.similarity,
    }
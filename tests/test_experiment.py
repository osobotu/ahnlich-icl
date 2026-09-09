import json

from ahnlich_icl.experiment import append_jsonl, build_generation_record
from ahnlich_icl.retrieval import SelectedExample
from ahnlich_icl.spider import Example
from ahnlich_icl.text2sql import SqlGeneration


def test_builds_generation_record():
    target = Example(
        example_id="dev:00001",
        question="How many singers are there?",
        sql="SELECT COUNT(*) FROM singer",
        db_id="concert_singer",
        difficulty="easy",
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
            source="similar",
            similarity=0.82,
        )
    ]
    generation = SqlGeneration(
        raw_output="```sql\nSELECT COUNT(*) FROM singer;\n```",
        sql="SELECT COUNT(*) FROM singer;",
    )

    record = build_generation_record(
        target,
        method="similar",
        selected_examples=selected,
        generation=generation,
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
                "difficulty": None,
                "source": "similar",
                "similarity": 0.82,
            }
        ],
        "raw_output": "```sql\nSELECT COUNT(*) FROM singer;\n```",
        "generated_sql": "SELECT COUNT(*) FROM singer;",
    }


def test_appends_one_json_object_per_line(tmp_path):
    output_path = tmp_path / "results" / "pilot.jsonl"
    first = {"example_id": "dev:00001"}
    second = {"example_id": "dev:00002"}

    append_jsonl(output_path, first)
    append_jsonl(output_path, second)

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert [json.loads(line) for line in lines] == [first, second]

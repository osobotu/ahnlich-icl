import argparse
import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from random import Random
from time import sleep

from ahnlich_icl.evaluation import SpiderEvaluator
from ahnlich_icl.experiment import (
    append_jsonl,
    build_generation_record,
    summarize_records,
)
from ahnlich_icl.retrieval import ExampleSelector, SelectionMethod
from ahnlich_icl.spider import Example, load_examples, load_schema
from ahnlich_icl.text2sql import generate_sql


DIFFICULTIES = ("easy", "medium", "hard", "extra")
METHODS: tuple[SelectionMethod, ...] = (
    "zero",
    "random",
    "similar",
)
PILOT_SEED = 2026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-difficulty", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/smoke.jsonl"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
    )
    return parser.parse_args()


def select_targets(
    examples: list[Example],
    evaluator: SpiderEvaluator,
    per_difficulty: int,
) -> list[Example]:
    buckets = {
        difficulty: []
        for difficulty in DIFFICULTIES
    }

    for example in examples:
        difficulty = evaluator.difficulty(example)
        buckets[difficulty].append(
            replace(example, difficulty=difficulty)
        )

    random = Random(PILOT_SEED)
    selected = []

    for difficulty in DIFFICULTIES:
        candidates = buckets[difficulty]
        if len(candidates) < per_difficulty:
            raise ValueError(
                f"Only {len(candidates)} {difficulty} examples available"
            )
        selected.extend(
            random.sample(candidates, per_difficulty)
        )

    return sorted(
        selected,
        key=lambda example: example.example_id,
    )


def main() -> None:
    args = parse_args()
    if args.per_difficulty <= 0:
        raise ValueError("--per-difficulty must be positive")

    if args.delay_seconds < 0:
        raise ValueError("--delay-seconds cannot be negative")

    spider_dir = Path(
        os.getenv("SPIDER_DIR", "data/spider")
    )
    store_name = os.getenv(
        "AHNLICH_STORE_NAME",
        "spider_examples",
    )

    training_examples = load_examples(spider_dir, "train")
    development_examples = load_examples(spider_dir, "dev")

    evaluator = SpiderEvaluator()
    selector = ExampleSelector(
        training_examples,
        store_name=store_name,
    )
    targets = select_targets(
        development_examples,
        evaluator,
        args.per_difficulty,
    )

    ids_path = args.output.with_suffix(".ids.json")
    summary_path = args.output.with_suffix(".summary.json")

    target_ids = [
        {
            "example_id": target.example_id,
            "difficulty": target.difficulty,
        }
        for target in targets
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        if not args.output.is_file() or not ids_path.is_file():
            raise FileNotFoundError(
                "--resume requires existing JSONL and ID files"
            )

        saved_ids = json.loads(
            ids_path.read_text(encoding="utf-8")
        )
        if saved_ids != target_ids:
            raise ValueError(
                "Saved pilot IDs do not match this run"
            )

        records = [
            json.loads(line)
            for line in args.output.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
    else:
        for path in (args.output, ids_path, summary_path):
            if path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing result: {path}"
                )

        ids_path.write_text(
            json.dumps(target_ids, indent=2),
            encoding="utf-8",
        )
        records = []

    completed_pairs = {
        (record["example_id"], record["method"])
        for record in records
    }
    if len(completed_pairs) != len(records):
        raise ValueError("Duplicate records found in output")

    total = len(targets) * len(METHODS)
    completed = len(records)

    print(f"Resuming from {completed}/{total} completed generations")

    for target in targets:
        pending_methods = [
            method
            for method in METHODS
            if (target.example_id, method)
            not in completed_pairs
        ]
        if not pending_methods:
            continue

        schema = load_schema(spider_dir, target.db_id)

        for method in pending_methods:
            selected_examples = asyncio.run(
                selector.select_examples(
                    method,
                    target,
                )
            )
            demonstrations = [
                selected.example
                for selected in selected_examples
            ]
            generated_sql = generate_sql(
                target.question,
                schema,
                demonstrations,
            )
            evaluation = evaluator.evaluate(
                target,
                generated_sql,
            )
            record = build_generation_record(
                target,
                method=method,
                selected_examples=selected_examples,
                generated_sql=generated_sql,
                evaluation=evaluation,
            )

            append_jsonl(args.output, record)
            records.append(record)

            completed += 1
            print(
                f"[{completed}/{total}] "
                f"{target.example_id} "
                f"{method}: "
                f"test_suite={evaluation.test_suite_correct} "
                f"exact={evaluation.exact_match}"
            )
            if completed < total:
                sleep(args.delay_seconds)

    summary = summarize_records(records)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

import pytest

from scripts.visualize_results import (
    group_complete_records,
    paired_counts,
)


def result(
    example_id: str,
    method: str,
    *,
    correct: bool,
) -> dict:
    return {
        "example_id": example_id,
        "method": method,
        "difficulty": "medium",
        "test_suite_correct": correct,
    }


def test_counts_paired_semantic_outcomes() -> None:
    outcomes = {
        "helped": (False, True, True),
        "hurt": (True, True, False),
        "both_correct": (True, False, True),
        "both_wrong": (False, False, False),
    }
    records = [
        result(example_id, method, correct=correct)
        for example_id, correctness in outcomes.items()
        for method, correct in zip(
            ("zero", "random", "similar"),
            correctness,
            strict=True,
        )
    ]

    grouped = group_complete_records(records)

    assert paired_counts(grouped, "zero") == {
        "helped": 1,
        "hurt": 1,
        "both_correct": 1,
        "both_wrong": 1,
    }
    assert paired_counts(grouped, "random") == {
        "helped": 1,
        "hurt": 1,
        "both_correct": 1,
        "both_wrong": 1,
    }


def test_rejects_incomplete_paired_results() -> None:
    records = [
        result("dev:00001", "zero", correct=True),
        result("dev:00001", "similar", correct=True),
    ]

    with pytest.raises(ValueError, match="Incomplete results"):
        group_complete_records(records)

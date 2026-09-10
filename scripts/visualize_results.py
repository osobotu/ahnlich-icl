import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ahnlich_icl.experiment import JsonRecord, summarize_records


METHODS = ("zero", "random", "similar")
METHOD_LABELS = {
    "zero": "Zero-shot",
    "random": "Random 5-shot",
    "similar": "Ahnlich similar 5-shot",
}
DIFFICULTIES = ("easy", "medium", "hard", "extra")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_records(path: Path) -> list[JsonRecord]:
    records = [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def group_complete_records(
    records: list[JsonRecord],
) -> dict[str, dict[str, JsonRecord]]:
    grouped: dict[str, dict[str, JsonRecord]] = {}

    for record in records:
        example_id = record["example_id"]
        method = record["method"]
        methods = grouped.setdefault(example_id, {})

        if method in methods:
            raise ValueError(
                f"Duplicate record: {example_id} {method}"
            )

        methods[method] = record

    expected_methods = set(METHODS)

    for example_id, methods in grouped.items():
        if set(methods) != expected_methods:
            raise ValueError(
                f"Incomplete results for {example_id}: "
                f"{sorted(methods)}"
            )

    return grouped


def paired_counts(
    grouped: dict[str, dict[str, JsonRecord]],
    baseline: str,
) -> dict[str, int]:
    counts = {
        "helped": 0,
        "hurt": 0,
        "both_correct": 0,
        "both_wrong": 0,
    }

    for methods in grouped.values():
        baseline_correct = methods[baseline][
            "test_suite_correct"
        ]
        similar_correct = methods["similar"][
            "test_suite_correct"
        ]

        if not baseline_correct and similar_correct:
            counts["helped"] += 1
        elif baseline_correct and not similar_correct:
            counts["hurt"] += 1
        elif baseline_correct:
            counts["both_correct"] += 1
        else:
            counts["both_wrong"] += 1

    return counts


def plot_results(
    records: list[JsonRecord],
    output: Path,
) -> None:
    grouped = group_complete_records(records)
    summary = summarize_records(records)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5),
        constrained_layout=True,
    )

    plot_overall(axes[0], summary)
    plot_by_difficulty(axes[1], summary)
    plot_transitions(axes[2], grouped)

    figure.suptitle(
        f"Spider Pilot Results — {len(grouped)} Questions"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_overall(axis, summary: JsonRecord) -> None:
    positions = range(len(METHODS))
    width = 0.36

    test_suite_bars = axis.bar(
        [position - width / 2 for position in positions],
        [
            summary[method]["test_suite_accuracy"]
            for method in METHODS
        ],
        width,
        label="Test suite",
    )
    exact_bars = axis.bar(
        [position + width / 2 for position in positions],
        [
            summary[method]["exact_match_accuracy"]
            for method in METHODS
        ],
        width,
        label="Exact match",
    )

    axis.bar_label(test_suite_bars, fmt="%.2f", fontsize=8)
    axis.bar_label(exact_bars, fmt="%.2f", fontsize=8)
    axis.set_title("Overall accuracy")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1.1)
    axis.set_xticks(
        list(positions),
        [METHOD_LABELS[method] for method in METHODS],
        rotation=15,
        ha="right",
    )
    axis.legend()


def plot_by_difficulty(axis, summary: JsonRecord) -> None:
    positions = range(len(DIFFICULTIES))
    width = 0.25

    for method_index, method in enumerate(METHODS):
        offset = (method_index - 1) * width
        axis.bar(
            [position + offset for position in positions],
            [
                summary[method]["by_difficulty"][difficulty][
                    "test_suite_accuracy"
                ]
                for difficulty in DIFFICULTIES
            ],
            width,
            label=METHOD_LABELS[method],
        )

    axis.set_title("Test-suite accuracy by difficulty")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1)
    axis.set_xticks(
        list(positions),
        ["Easy", "Medium", "Hard", "Extra hard"],
    )
    axis.legend(fontsize=8)


def plot_transitions(
    axis,
    grouped: dict[str, dict[str, JsonRecord]],
) -> None:
    outcomes = (
        "helped",
        "hurt",
        "both_correct",
        "both_wrong",
    )
    labels = (
        "Helped",
        "Hurt",
        "Both correct",
        "Both wrong",
    )
    positions = range(len(outcomes))
    width = 0.36

    versus_zero = paired_counts(grouped, "zero")
    versus_random = paired_counts(grouped, "random")

    zero_bars = axis.bar(
        [position - width / 2 for position in positions],
        [versus_zero[outcome] for outcome in outcomes],
        width,
        label="Similar vs zero",
    )
    random_bars = axis.bar(
        [position + width / 2 for position in positions],
        [versus_random[outcome] for outcome in outcomes],
        width,
        label="Similar vs random",
    )

    axis.bar_label(zero_bars, fontsize=8)
    axis.bar_label(random_bars, fontsize=8)
    axis.set_title("Paired semantic outcomes")
    axis.set_ylabel("Questions")
    axis.set_xticks(
        list(positions),
        labels,
        rotation=15,
        ha="right",
    )
    axis.legend(fontsize=8)


def main() -> None:
    args = parse_args()
    output = args.output or args.input.with_suffix(".png")

    records = load_records(args.input)
    plot_results(records, output)

    print(f"Saved visualization to {output}")


if __name__ == "__main__":
    main()
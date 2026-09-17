import asyncio

import ahnlich_icl.retrieval as retrieval
from ahnlich_icl.ahnlich_mcp import SearchMatch
from ahnlich_icl.spider import Example


def make_example(example_id: str) -> Example:
    return Example(
        example_id=example_id,
        question=f"Question {example_id}",
        sql=f"SELECT '{example_id}'",
        db_id="database",
    )


def make_selector(examples: list[Example]) -> retrieval.ExampleSelector:
    return retrieval.ExampleSelector(
        examples,
        store_name="spider_examples",
    )


def test_zero_shot_selects_nothing():
    examples = [make_example(f"train:{number:05d}") for number in range(1, 11)]
    selector = make_selector(examples)

    selected = asyncio.run(
        selector.select_examples("zero", make_example("dev:00001"), k=5)
    )

    assert selected == []


def test_random_selection_is_reproducible_and_order_independent():
    examples = [make_example(f"train:{number:05d}") for number in range(1, 11)]
    target = make_example("dev:00001")

    first = asyncio.run(
        make_selector(examples).select_examples("random", target, k=5)
    )
    second = asyncio.run(
        make_selector(list(reversed(examples))).select_examples(
            "random", target, k=5
        )
    )

    assert [item.example.example_id for item in first] == [
        item.example.example_id for item in second
    ]


def test_similar_selection_recovers_examples_and_scores(monkeypatch):
    examples = [make_example(f"train:{number:05d}") for number in range(1, 11)]
    selector = make_selector(examples)

    async def search(store_name: str, question: str, k: int):
        assert (store_name, question, k) == (
            "spider_examples",
            "Target question",
            2,
        )
        return [
            SearchMatch(
                Example(
                    "train:00006",
                    "Retrieved one",
                    "SELECT 1",
                    "first_database",
                ),
                similarity=0.9,
            ),
            SearchMatch(
                Example(
                    "train:00007",
                    "Retrieved two",
                    "SELECT 2",
                    "second_database",
                ),
                similarity=0.8,
            ),
        ]

    monkeypatch.setattr(retrieval, "similarity_search", search)
    target = Example(
        example_id="dev:00001",
        question="Target question",
        sql="SELECT 1",
        db_id="target_database",
    )

    selected = asyncio.run(selector.select_examples("similar", target, k=2))

    assert [item.example.question for item in selected] == [
        "Retrieved one",
        "Retrieved two",
    ]
    assert [item.similarity for item in selected] == [0.9, 0.8]

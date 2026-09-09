from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from random import Random
from typing import Literal

from ahnlich_icl.ahnlich_mcp import similarity_search
from ahnlich_icl.spider import Example


SelectionMethod = Literal[
    "zero",
    "random",
    "similar",
]
SelectionSource = Literal["random", "similar"]

RANDOM_SEED = 7

@dataclass(frozen=True)
class SelectedExample:
    example: Example
    source: SelectionSource
    similarity: float | None = None


class ExampleSelector:
    def __init__(
        self,
        training_examples: Sequence[Example],
        *,
        store_name: str,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        self._training_examples = tuple(
            sorted(training_examples, key=lambda example: example.example_id)
        )
        self._store_name = store_name
        self._random_seed = random_seed

    async def select_examples(
        self,
        method: SelectionMethod,
        target: Example,
        k: int = 5,
    ) -> list[SelectedExample]:
        if k <= 0:
            raise ValueError("k must be positive")

        if method == "zero":
            return []
        if method == "random":
            return self._select_random(target, k)
        if method == "similar":
            return await self._select_similar(target, k)

        raise ValueError(f"Unknown selection method: {method}")

    def _select_random(
        self,
        target: Example,
        k: int,
    ) -> list[SelectedExample]:
        candidates = [
            example
            for example in self._training_examples
            if example.example_id != target.example_id
        ]
        if len(candidates) < k:
            raise ValueError(f"Cannot select {k} random examples")

        seed_text = f"{self._random_seed}:{target.example_id}"
        seed_bytes = sha256(seed_text.encode()).digest()
        random = Random(int.from_bytes(seed_bytes[:8], "big"))

        return [
            SelectedExample(example=example, source="random")
            for example in random.sample(candidates, k)
        ]

    async def _select_similar(
        self,
        target: Example,
        k: int,
    ) -> list[SelectedExample]:
        matches = await similarity_search(
            self._store_name,
            target.question,
            k,
        )
        if len(matches) != k:
            raise RuntimeError(
                f"Ahnlich returned {len(matches)} matches; expected {k}"
            )

        return [
            SelectedExample(
                example=match.example,
                source="similar",
                similarity=match.similarity,
            )
            for match in matches
        ]
import asyncio
import os
from pathlib import Path
from time import monotonic

from ahnlich_icl.ahnlich_mcp import (
    create_store,
    drop_store,
    store_examples,
)
from ahnlich_icl.spider import load_examples


BATCH_SIZE = 100


async def main() -> None:
    spider_dir = Path(os.getenv("SPIDER_DIR", "data/spider"))
    store_name = os.getenv("AHNLICH_STORE_NAME", "spider_examples")
    examples = load_examples(spider_dir, "train")

    await drop_store(store_name)
    await create_store(store_name)

    inserted = 0
    updated = 0
    failed = 0
    started = monotonic()

    for start in range(0, len(examples), BATCH_SIZE):
        batch = examples[start : start + BATCH_SIZE]

        try:
            result = await store_examples(store_name, batch)
            inserted += result["inserted"]
            updated += result["updated"]
        except Exception as error:
            failed += len(batch)
            print(f"Batch starting at {start} failed: {error}")

        print(f"Processed {min(start + BATCH_SIZE, len(examples))}/{len(examples)}")

    print()
    print(f"Attempted: {len(examples)}")
    print(f"Inserted:  {inserted}")
    print(f"Updated:   {updated}")
    print(f"Failed:    {failed}")
    print(f"Elapsed:   {monotonic() - started:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
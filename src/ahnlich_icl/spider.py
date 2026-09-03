import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


SPLIT_FILES = {
    "train": "train_spider.json",
    "dev": "dev.json",
}


@dataclass(frozen=True)
class Example:
    example_id: str
    question: str
    sql: str
    db_id: str
    difficulty: str | None = None


def load_examples(spider_dir: Path, split: str) -> list[Example]:
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown Spider split: {split}")

    split_path = spider_dir / SPLIT_FILES[split]
    if not split_path.is_file():
        raise FileNotFoundError(f"Spider split not found: {split_path}")

    rows = json.loads(split_path.read_text(encoding="utf-8"))

    return [
        Example(
            example_id=f"{split}:{index:05d}",
            question=row["question"],
            sql=row["query"],
            db_id=row["db_id"],
        )
        for index, row in enumerate(rows, start=1)
    ]


def database_path(spider_dir: Path, db_id: str) -> Path:
    path = spider_dir / "database" / db_id / f"{db_id}.sqlite"
    if not path.is_file():
        raise FileNotFoundError(f"Spider database not found: {path}")
    return path


def load_schema(spider_dir: Path, db_id: str) -> str:
    path = database_path(spider_dir, db_id)

    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

    definitions = [row[0] for row in rows if row[0]]
    if not definitions:
        raise ValueError(f"No tables found in Spider database: {db_id}")

    return "\n\n".join(
        f"{definition.rstrip().rstrip(';')};"
        for definition in definitions
    )
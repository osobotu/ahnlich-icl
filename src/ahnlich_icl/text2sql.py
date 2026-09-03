from ahnlich_icl.spider import Example


SYSTEM_PROMPT = (
    "Translate the question into SQLite SQL using the provided database "
    "schema. Return exactly one SQL query with no explanation or Markdown."
)


def build_prompt(
    question: str,
    schema: str,
    demonstrations: list[Example],
) -> str:
    sections = [
        "DATABASE SCHEMA",
        schema.strip(),
    ]

    for number, example in enumerate(demonstrations, start=1):
        sections.extend(
            [
                f"EXAMPLE {number}",
                f"Question:\n{example.question}",
                f"SQL:\n{example.sql}",
            ]
        )

    sections.extend(
        [
            "TARGET QUESTION",
            question,
            "SQL:",
        ]
    )

    return "\n\n".join(sections)
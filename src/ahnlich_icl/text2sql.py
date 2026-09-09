import os
from collections.abc import Sequence
from functools import cache
from dataclasses import dataclass

from openai import OpenAI

from ahnlich_icl.spider import Example


SYSTEM_PROMPT = (
    "Translate the question into SQLite SQL using the provided database "
    "schema. Return exactly one SQL query with no explanation or Markdown."
)
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"
MAX_COMPLETION_TOKENS = 512

@dataclass(frozen=True)
class SqlGeneration:
    raw_output: str
    sql: str


def build_prompt(
    question: str,
    schema: str,
    demonstrations: Sequence[Example] = (),
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

def generate_sql(
    question: str,
    schema: str,
    demonstrations: Sequence[Example] = (),
) -> SqlGeneration:
    response = _llm_client().chat.completions.create(
        model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
        temperature=0,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_prompt(question, schema, demonstrations),
            },
        ],
    )

    raw_output = response.choices[0].message.content
    if not raw_output:
        raise RuntimeError("LLM returned an empty response")

    return SqlGeneration(
        raw_output=raw_output,
        sql=extract_sql(raw_output),
    )

def extract_sql(raw_output: str) -> str:
    stripped_output = raw_output.strip()
    lines = stripped_output.splitlines()

    is_fenced = (
        len(lines) >= 2
        and lines[0].strip().startswith("```")
        and lines[-1].strip() == "```"
    )
    if is_fenced:
        return "\n".join(lines[1:-1]).strip()

    return stripped_output

@cache
def _llm_client() -> OpenAI:
    return OpenAI(
        api_key=_api_key(),
        base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
    )


def _api_key() -> str:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set LLM_API_KEY or GROQ_API_KEY")

    return api_key
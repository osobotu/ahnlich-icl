from pathlib import Path
from types import SimpleNamespace

import ahnlich_icl.evaluation as evaluation_module
from ahnlich_icl.evaluation import SpiderEvaluator
from ahnlich_icl.spider import Example


class FakeEvaluator:
    def eval_hardness(self, sql: dict) -> str:
        return "easy"

    def eval_exact_match(self, predicted: dict, gold: dict) -> int:
        return int(predicted["text"] == gold["text"])


def make_official_module(execution_calls: list[dict]) -> SimpleNamespace:
    def parse_sql(schema: object, sql: str) -> dict:
        return {"text": sql, "from": {"table_units": []}}

    def evaluate_execution(**arguments: object) -> int:
        execution_calls.append(arguments)
        return 1

    return SimpleNamespace(
        Evaluator=FakeEvaluator,
        Schema=lambda schema: schema,
        build_foreign_key_map_from_json=lambda path: {"company": {}},
        build_valid_col_units=lambda tables, schema: [],
        eval_exec_match=evaluate_execution,
        get_schema=lambda path: {},
        get_sql=parse_sql,
        rebuild_sql_col=lambda columns, sql, key_map: sql,
        rebuild_sql_val=lambda sql: sql,
    )


def make_spider_evaluator(
    tmp_path: Path,
    monkeypatch,
    execution_calls: list[dict],
) -> SpiderEvaluator:
    database_path = (
        tmp_path
        / "data/test-suite/database/company/company.sqlite"
    )
    database_path.parent.mkdir(parents=True)
    database_path.touch()
    official_module = make_official_module(execution_calls)
    monkeypatch.setattr(
        evaluation_module,
        "_load_official_evaluator",
        lambda path: official_module,
    )
    return SpiderEvaluator(tmp_path)


def example(sql: str = "SELECT 1") -> Example:
    return Example(
        example_id="dev:00001",
        question="Return one.",
        sql=sql,
        db_id="company",
    )


def test_gold_prediction_passes_both_official_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execution_calls: list[dict] = []
    evaluator = make_spider_evaluator(
        tmp_path,
        monkeypatch,
        execution_calls,
    )

    result = evaluator.evaluate(example(), "SELECT 1")

    assert result.difficulty == "easy"
    assert result.test_suite_correct is True
    assert result.exact_match is True
    assert result.error is None
    assert execution_calls[0]["plug_value"] is False
    assert execution_calls[0]["keep_distinct"] is False
    assert execution_calls[0]["progress_bar_for_each_datapoint"] is False


def test_derives_difficulty_without_running_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execution_calls: list[dict] = []
    evaluator = make_spider_evaluator(
        tmp_path,
        monkeypatch,
        execution_calls,
    )

    assert evaluator.difficulty(example()) == "easy"
    assert execution_calls == []


def test_execution_and_exact_match_remain_separate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evaluator = make_spider_evaluator(tmp_path, monkeypatch, [])

    result = evaluator.evaluate(example(), "SELECT 2")

    assert result.test_suite_correct is True
    assert result.exact_match is False
    assert result.error is None


def test_invalid_prediction_becomes_a_structured_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evaluator = make_spider_evaluator(tmp_path, monkeypatch, [])

    result = evaluator.evaluate(example(), "SELECT missing_column")

    assert result.test_suite_correct is False
    assert result.exact_match is False
    assert "missing_column" in result.error


def test_mutating_prediction_never_reaches_official_evaluator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execution_calls: list[dict] = []
    evaluator = make_spider_evaluator(
        tmp_path,
        monkeypatch,
        execution_calls,
    )

    result = evaluator.evaluate(example(), "CREATE TABLE changed (id INTEGER)")

    assert result.test_suite_correct is False
    assert result.exact_match is False
    assert result.error is not None
    assert execution_calls == []

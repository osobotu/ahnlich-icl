import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
import importlib.util
import sys
from types import ModuleType

from ahnlich_icl.spider import Example


_PROGRESS_HANDLER_STEPS = 10_000


@dataclass(frozen=True)
class ExecutionResult:
    execution_match: bool
    error: str | None = None


def evaluate_execution(
    database_path: Path,
    predicted_sql: str,
    gold_sql: str,
    *,
    timeout_seconds: float = 5.0,
) -> ExecutionResult:
    """Compare result-row multisets without modifying the database."""
    try:
        gold_rows = _execute_query(
            database_path,
            gold_sql,
            timeout_seconds,
        )
    except sqlite3.Error as error:
        raise RuntimeError(f"Gold SQL failed: {error}") from error

    try:
        predicted_rows = _execute_query(
            database_path,
            predicted_sql,
            timeout_seconds,
        )
    except sqlite3.Error as error:
        return ExecutionResult(
            execution_match=False,
            error=str(error),
        )

    return ExecutionResult(
        execution_match=Counter(predicted_rows) == Counter(gold_rows)
    )


def _execute_query(
    database_path: Path,
    sql: str,
    timeout_seconds: float,
) -> list[tuple[object, ...]]:
    deadline = monotonic() + timeout_seconds
    database_uri = database_path.resolve().as_uri() + "?mode=ro"

    with sqlite3.connect(database_uri, uri=True) as connection:
        connection.set_progress_handler(
            lambda: monotonic() >= deadline,
            _PROGRESS_HANDLER_STEPS,
        )
        return connection.execute(sql).fetchall()

@dataclass(frozen=True)
class SpiderEvaluation:
    difficulty: str | None
    test_suite_correct: bool
    exact_match: bool
    error: str | None = None


class SpiderEvaluator:
    def __init__(self, project_root: Path = Path(".")) -> None:
        evaluator_dir = project_root / "tools/test-suite-sql-eval"
        tables_path = project_root / "data/spider/tables.json"

        self._official = _load_official_evaluator(evaluator_dir)
        self._database_dir = project_root / "data/test-suite/database"
        self._foreign_keys = (
            self._official.build_foreign_key_map_from_json(
                str(tables_path)
            )
        )

    def evaluate(
        self,
        example: Example,
        predicted_sql: str,
    ) -> SpiderEvaluation:
        try:
            return self._evaluate_one(example, predicted_sql)
        except Exception as error:
            return _failed_evaluation(None, _error_text(error))

    def difficulty(self, example: Example) -> str:
        database_path = self._database_path(example.db_id)
        schema = self._official.Schema(
            self._official.get_schema(str(database_path))
        )
        gold_tree = self._official.get_sql(schema, example.sql)

        return self._official.Evaluator().eval_hardness(gold_tree)

    def _evaluate_one(
        self,
        example: Example,
        predicted_sql: str,
    ) -> SpiderEvaluation:
        database_path = self._database_path(example.db_id)
        if not database_path.is_file():
            raise FileNotFoundError(
                f"Test-suite database not found: {database_path}"
            )

        schema = self._official.Schema(
            self._official.get_schema(str(database_path))
        )
        gold_tree = self._official.get_sql(schema, example.sql)
        difficulty = self._official.Evaluator().eval_hardness(
            gold_tree
        )

        preflight = evaluate_execution(
            database_path,
            predicted_sql,
            example.sql,
        )
        if preflight.error is not None:
            return _failed_evaluation(
                difficulty,
                preflight.error,
            )

        test_suite_correct, execution_error = (
            self._test_suite_match(
                database_path,
                predicted_sql,
                example.sql,
            )
        )
        if execution_error is not None:
            return _failed_evaluation(
                difficulty,
                execution_error,
            )

        exact_match, exact_error = self._exact_match(
            example,
            schema,
            predicted_sql,
        )

        return SpiderEvaluation(
            difficulty=difficulty,
            test_suite_correct=test_suite_correct,
            exact_match=exact_match,
            error=exact_error,
        )

    def _database_path(self, db_id: str) -> Path:
        database_path = (
            self._database_dir
            / db_id
            / f"{db_id}.sqlite"
        )
        if not database_path.is_file():
            raise FileNotFoundError(
                f"Test-suite database not found: {database_path}"
            )

        return database_path

    def _test_suite_match(
        self,
        database_path: Path,
        predicted_sql: str,
        gold_sql: str,
    ) -> tuple[bool, str | None]:
        try:
            score = self._official.eval_exec_match(
                db=str(database_path),
                p_str=predicted_sql,
                g_str=gold_sql,
                plug_value=False,
                keep_distinct=False,
                progress_bar_for_each_datapoint=False,
            )
            return bool(score), None
        except Exception as error:
            return False, _error_text(error)

    def _exact_match(
        self,
        example: Example,
        schema: object,
        predicted_sql: str,
    ) -> tuple[bool, str | None]:
        try:
            key_map = self._foreign_keys[example.db_id]
            predicted_tree = self._normalize_sql(
                schema,
                predicted_sql,
                key_map,
            )
            gold_tree = self._normalize_sql(
                schema,
                example.sql,
                key_map,
            )
            score = self._official.Evaluator().eval_exact_match(
                predicted_tree,
                gold_tree,
            )
            return bool(score), None
        except Exception as error:
            return False, _error_text(error)

    def _normalize_sql(
        self,
        schema: object,
        sql: str,
        key_map: dict,
    ) -> dict:
        parsed_sql = self._official.get_sql(schema, sql)
        valid_columns = self._official.build_valid_col_units(
                       parsed_sql["from"]["table_units"],
            schema,
        )
        parsed_sql = self._official.rebuild_sql_val(parsed_sql)

        return self._official.rebuild_sql_col(
            valid_columns,
            parsed_sql,
            key_map,
        )


def _failed_evaluation(
    difficulty: str | None,
    error: str,
) -> SpiderEvaluation:
    return SpiderEvaluation(
        difficulty=difficulty,
        test_suite_correct=False,
        exact_match=False,
        error=error,
    )


def _error_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _load_official_evaluator(
    evaluator_dir: Path,
) -> ModuleType:
    evaluator_path = evaluator_dir / "evaluation.py"
    if not evaluator_path.is_file():
        raise FileNotFoundError(
            f"Official Spider evaluator not found: {evaluator_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "_official_spider_evaluation",
        evaluator_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load Spider evaluator: {evaluator_path}"
        )

    module    = importlib.util.module_from_spec(spec)

    # The upstream evaluator uses un-packaged sibling imports.
    sys.path.insert(0, str(evaluator_dir.resolve()))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    return module

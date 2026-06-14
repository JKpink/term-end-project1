"""Tests for evaluation engine."""
import pytest
from src.evaluator import (
    compare_result_sets,
    normalize_sql,
    _values_equal,
    score_analysis_question,
)


class TestValuesEqual:
    def test_both_none(self):
        assert _values_equal(None, None)

    def test_one_none(self):
        assert not _values_equal(None, 5)
        assert not _values_equal(5, None)

    def test_integers_equal(self):
        assert _values_equal(42, 42)

    def test_integers_not_equal(self):
        assert not _values_equal(42, 43)

    def test_float_tolerance(self):
        assert _values_equal(1.0001, 1.0000)
        assert _values_equal(1.009, 1.000)  # 0.009 < 0.01

    def test_float_outside_tolerance(self):
        assert not _values_equal(1.02, 1.00)

    def test_string_comparison(self):
        assert _values_equal("hello", "hello")
        assert not _values_equal("hello", "world")


class TestCompareResultSets:
    def test_identical(self):
        pred = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        gold = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        assert compare_result_sets(pred, gold)

    def test_different_order(self):
        pred = [{"a": 2, "b": "y"}, {"a": 1, "b": "x"}]
        gold = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        assert compare_result_sets(pred, gold)  # sorted comparison

    def test_different_row_count(self):
        pred = [{"a": 1}]
        gold = [{"a": 1}, {"a": 2}]
        assert not compare_result_sets(pred, gold)

    def test_different_column_count(self):
        pred = [{"a": 1, "b": 2}]
        gold = [{"a": 1}]
        assert not compare_result_sets(pred, gold)

    def test_null_matching(self):
        pred = [{"a": None, "b": "x"}]
        gold = [{"a": None, "b": "x"}]
        assert compare_result_sets(pred, gold)

    def test_null_vs_value(self):
        pred = [{"a": None}]
        gold = [{"a": 0}]
        assert not compare_result_sets(pred, gold)

    def test_float_tolerance_in_result(self):
        pred = [{"val": 3.14159}]
        gold = [{"val": 3.14158}]
        assert compare_result_sets(pred, gold)

    def test_empty_results(self):
        assert compare_result_sets([], [])


class TestNormalizeSQL:
    def test_lowercase(self):
        assert normalize_sql("SELECT * FROM Users") == "select * from users"

    def test_strip_semicolon(self):
        s = normalize_sql("SELECT * FROM users;")
        assert ";" not in s

    def test_collapse_whitespace(self):
        s = normalize_sql("SELECT   *\nFROM    users\nWHERE  id=1")
        assert "  " not in s

    def test_normalize_quotes(self):
        s = normalize_sql('SELECT * FROM "users"')
        assert "'" in s  # double → single quotes


class TestScoreAnalysisQuestion:
    def test_perfect_score(self):
        result = {
            "sql": "SELECT dept, COUNT(*) FROM users GROUP BY dept",
            "error": None,
            "rows": [{"dept": "CS", "count": 10}, {"dept": "Math", "count": 5}],
            "chart_type": "bar",
            "report": "## 概述\n分析发现CS学院最活跃\n## 细节\n交易量增长了30%",
            "findings": ["CS学院排名第一", "占总交易量45%"],
        }
        ground_truth = {
            "required_elements": ["CS", "Math", "排名"],
            "expected_chart_type": "bar",
            "min_report_sections": 2,
        }
        scores = score_analysis_question(result, ground_truth)
        assert scores["sql_exec"] == 1.0
        assert scores["chart_ok"] == 1.0
        assert scores["report_ok"] >= 0.5
        assert "weighted_total" in scores

    def test_no_sql(self):
        result = {"sql": "", "error": "No SQL", "rows": [], "chart_type": "", "report": ""}
        ground_truth = {"required_elements": [], "expected_chart_type": "", "min_report_sections": 1}
        scores = score_analysis_question(result, ground_truth)
        assert scores["sql_exec"] == 0.0

    def test_sql_with_error(self):
        result = {"sql": "BROKEN", "error": "syntax error", "rows": [], "chart_type": "", "report": ""}
        ground_truth = {"required_elements": [], "expected_chart_type": "", "min_report_sections": 1}
        scores = score_analysis_question(result, ground_truth)
        assert scores["sql_exec"] == 0.5

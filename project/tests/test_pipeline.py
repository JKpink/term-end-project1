"""Integration tests for pipeline with mock LLM."""
import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.datasource import SQLiteDataSource
from src.pipeline import run_single_question

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "campus_trade.db")


@pytest.fixture
def db():
    return SQLiteDataSource(DB_PATH)


@pytest.fixture
def mock_llm_chat():
    """Mock LLMClient.chat to return fake SQL for all agents."""
    with patch("src.llm_client.LLMClient.chat") as mock:
        def side_effect(messages, **kwargs):
            # Return a SQL query for any call
            return '{"sql": "SELECT COUNT(*) as cnt FROM users", "explanation": "test"}'
        mock.side_effect = side_effect
        yield mock


def test_pipeline_b1_with_mock(db, mock_llm_chat):
    """B1: Single Direct should work with mocked LLM."""
    results = run_single_question(
        "How many users?",
        db,
        methods=["B1"],
    )
    assert "B1" in results
    r = results["B1"]
    # With mock, should get a valid SQL
    assert r.get("error") is None or "API" not in str(r.get("error", ""))
    # At minimum, the pipeline should complete without crashing


def test_pipeline_all_methods_with_mock(db, mock_llm_chat):
    """All 8 methods should complete without unhandled exceptions."""
    methods = ["B1", "B2", "B3", "B4", "Ours",
               "Ours_no_evolution", "Ours_no_experience", "Ours_no_split"]
    results = run_single_question("How many users are there?", db, methods=methods)
    assert len(results) == 8
    for method, r in results.items():
        assert "elapsed" in r, f"{method} missing elapsed"
        assert "method" in r, f"{method} missing method field"


def test_pipeline_error_handling(db):
    """Pipeline should handle errors gracefully without crashing."""
    # Make the mock fail
    with patch("src.llm_client.LLMClient.chat", side_effect=RuntimeError("Simulated failure")):
        results = run_single_question("test", db, methods=["B1", "Ours"])
        # Should return error entries, not crash
        assert "B1" in results
        assert results["B1"].get("error") is not None


def test_pipeline_returns_expected_keys(db, mock_llm_chat):
    """Each method should return the expected fields."""
    results = run_single_question("SELECT question", db, methods=["B1", "Ours"])

    # B1 keys
    b1 = results["B1"]
    for key in ["method", "sql", "rows", "error", "elapsed"]:
        assert key in b1, f"B1 missing {key}"

    # Ours keys
    ours = results["Ours"]
    for key in ["method", "sql", "rows", "error", "elapsed", "trace"]:
        assert key in ours, f"Ours missing {key}"

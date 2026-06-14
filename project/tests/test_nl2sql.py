"""Tests for NL2SQL engine (with mock LLM)."""
import pytest
from unittest.mock import MagicMock, patch
from src.nl2sql import NL2SQLEngine


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    return llm


@pytest.fixture
def engine(mock_llm):
    return NL2SQLEngine(llm=mock_llm)


def test_generate_sql_clean_output(engine, mock_llm):
    mock_llm.chat.return_value = "SELECT * FROM users"
    sql = engine.generate_sql("how many users?", "CREATE TABLE users (...)")
    assert sql == "SELECT * FROM users"


def test_generate_sql_strips_markdown(engine, mock_llm):
    mock_llm.chat.return_value = "```sql\nSELECT * FROM users\n```"
    sql = engine.generate_sql("question", "schema")
    assert sql == "SELECT * FROM users"


def test_generate_sql_strips_semicolon(engine, mock_llm):
    mock_llm.chat.return_value = "SELECT * FROM users;"
    sql = engine.generate_sql("question", "schema")
    assert sql == "SELECT * FROM users"


def test_generate_sql_strips_code_fence_no_lang(engine, mock_llm):
    mock_llm.chat.return_value = "```\nSELECT COUNT(*) FROM products\n```"
    sql = engine.generate_sql("question", "schema")
    assert sql == "SELECT COUNT(*) FROM products"


def test_generate_sql_handles_complex_output(engine, mock_llm):
    mock_llm.chat.return_value = "SELECT u.name, COUNT(o.id)\nFROM users u\nJOIN orders o ON u.id = o.buyer_id\nGROUP BY u.name;"
    sql = engine.generate_sql("question", "schema")
    assert "SELECT" in sql
    assert "JOIN" in sql
    assert ";" not in sql


def test_clean_sql_multiline_fence(engine):
    """Test _clean_sql directly with multiline markdown fence."""
    result = engine._clean_sql("```sql\nSELECT a,\n  b\nFROM t\n```")
    assert result == "SELECT a,\n  b\nFROM t"

"""Tests for SQLiteDataSource."""
import os
import pytest
from src.datasource import SQLiteDataSource

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "campus_trade.db")


@pytest.fixture
def db():
    return SQLiteDataSource(DB_PATH)


def test_db_exists():
    assert os.path.exists(DB_PATH), f"Test DB not found: {DB_PATH}"


def test_get_table_names(db):
    tables = db.get_table_names()
    assert "users" in tables
    assert "products" in tables
    assert "orders" in tables
    assert "reviews" in tables
    assert len(tables) >= 6


def test_get_schema_text(db):
    schema = db.get_schema_text()
    assert "CREATE TABLE users" in schema
    assert "CREATE TABLE products" in schema
    assert "Sample rows" in schema  # sample data included
    assert len(schema) > 1000


def test_execute_valid_query(db):
    rows, error = db.execute_query("SELECT COUNT(*) as cnt FROM users")
    assert error is None
    assert len(rows) == 1
    assert rows[0]["cnt"] >= 1


def test_execute_invalid_query(db):
    rows, error = db.execute_query("SELECT * FROM nonexistent_table")
    assert error is not None
    assert rows == []
    assert "no such table" in error.lower()


def test_execute_gold(db):
    rows = db.execute_gold("SELECT COUNT(*) as cnt FROM products WHERE campus = '主校区'")
    assert len(rows) == 1
    assert rows[0]["cnt"] >= 0


def test_null_handling(db):
    """Verify NULL values are properly returned from queries."""
    # Use complete_time which can be NULL in orders table
    rows = db.execute_gold("SELECT id, complete_time FROM orders WHERE complete_time IS NULL LIMIT 5")
    for row in rows:
        assert row["complete_time"] is None
        assert row["id"] is not None


def test_multiple_tables_query(db):
    rows = db.execute_gold(
        "SELECT u.nickname, COUNT(p.id) as cnt "
        "FROM users u LEFT JOIN products p ON u.id = p.seller_id "
        "WHERE p.status = 'active' "
        "GROUP BY u.id HAVING cnt > 0 LIMIT 5"
    )
    assert len(rows) <= 5
    if rows:
        assert "nickname" in rows[0]
        assert "cnt" in rows[0]

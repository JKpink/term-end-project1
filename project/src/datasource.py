"""SQLite data source adapter — schema extraction, query execution."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Optional


class SQLiteDataSource:
    """Wraps a SQLite database, provides schema extraction and safe query execution."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def get_schema_text(self) -> str:
        """Extract full schema as human-readable text for LLM prompting."""
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [r[0] for r in cur.fetchall()]

        lines = []
        for table in tables:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(f"PRAGMA table_info('{table}')")
                cols = cur.fetchall()
                # sample rows
                try:
                    cur.execute(f"SELECT * FROM '{table}' LIMIT 3")
                    samples = cur.fetchall()
                except sqlite3.Error:
                    samples = []

            col_defs = []
            for c in cols:
                pk = " PRIMARY KEY" if c[5] else ""
                nullable = "" if c[3] else " NOT NULL"
                col_defs.append(f"  {c[1]} {c[2]}{pk}{nullable}")
            lines.append(f"CREATE TABLE {table} (")
            lines.append(",\n".join(col_defs))
            lines.append(");")

            if samples:
                lines.append(f"-- Sample rows ({min(3, len(samples))}):")
                for row in samples:
                    lines.append(f"--   {dict(row)}")
            lines.append("")

        return "\n".join(lines)

    def get_table_names(self) -> list[str]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            return [r[0] for r in cur.fetchall()]

    def execute_query(self, sql: str) -> tuple[list[dict[str, Any]], Optional[str]]:
        """Execute a SELECT query. Returns (rows, error_message)."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(sql)
                rows = cur.fetchall()
                return [dict(r) for r in rows], None
        except sqlite3.Error as e:
            return [], str(e)

    def execute_gold(self, sql: str) -> list[dict[str, Any]]:
        """Execute gold SQL, raise on error."""
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(sql)
            return [dict(r) for r in cur.fetchall()]

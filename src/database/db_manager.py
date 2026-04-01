"""
Database manager with multi-dialect support via SQLAlchemy.

Provides:
  - DatabaseManager  – new API used by RLLoop, demo, and tests
  - ExecutionResult  – structured result dataclass
  - DBManager        – legacy thin wrapper kept for tasks.py compatibility
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import List, Optional

import sqlalchemy as sa
from sqlalchemy import inspect as sa_inspect, text
from sqlalchemy.pool import StaticPool

MAX_ROWS = 1000

_UNSAFE_PATTERN = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|TRUNCATE|ALTER|CREATE|REPLACE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)


def _is_safe_sql(sql: str) -> bool:
    """Return True only if the SQL contains no write/DDL keywords."""
    return not bool(_UNSAFE_PATTERN.search(sql))


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

@dataclass
class ColumnInfo:
    name: str
    type: str


@dataclass
class TableSchema:
    name: str
    columns: List[ColumnInfo]


@dataclass
class DatabaseSchema:
    tables: List[TableSchema]


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    success: bool
    rows: List[tuple] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    error: Optional[str] = None

    def to_display(self) -> str:
        if not self.success:
            return f"Error: {self.error}"
        if not self.rows:
            return "(no rows returned)"
        header = " | ".join(self.columns)
        separator = "-" * max(len(header), 1)
        rows_str = "\n".join(" | ".join(str(c) for c in row) for row in self.rows[:20])
        suffix = f"\n... ({self.row_count} total rows)" if self.row_count > 20 else ""
        return f"{header}\n{separator}\n{rows_str}{suffix}"


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

_SAMPLE_DDL = [
    """CREATE TABLE IF NOT EXISTS employees (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        name    TEXT    NOT NULL,
        dept    TEXT    NOT NULL,
        salary  REAL    NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS departments (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        name    TEXT    NOT NULL,
        budget  REAL    NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS projects (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        name      TEXT NOT NULL,
        end_date  TEXT
    )""",
]

_SAMPLE_INSERTS = {
    "employees": (
        "INSERT INTO employees (name, dept, salary) VALUES (:name, :dept, :salary)",
        [
            {"name": "Alice",   "dept": "Engineering", "salary": 95000},
            {"name": "Bob",     "dept": "Marketing",   "salary": 72000},
            {"name": "Charlie", "dept": "Engineering", "salary": 88000},
            {"name": "Diana",   "dept": "HR",          "salary": 65000},
            {"name": "Eve",     "dept": "Marketing",   "salary": 78000},
            {"name": "Frank",   "dept": "Engineering", "salary": 102000},
        ],
    ),
    "departments": (
        "INSERT INTO departments (name, budget) VALUES (:name, :budget)",
        [
            {"name": "Engineering", "budget": 500000},
            {"name": "Marketing",   "budget": 200000},
            {"name": "HR",          "budget": 150000},
        ],
    ),
    "projects": (
        "INSERT INTO projects (name, end_date) VALUES (:name, :end_date)",
        [
            {"name": "Alpha", "end_date": "2024-03-01"},
            {"name": "Beta",  "end_date": None},
            {"name": "Gamma", "end_date": "2025-01-15"},
        ],
    ),
}


# ---------------------------------------------------------------------------
# DatabaseManager
# ---------------------------------------------------------------------------

class DatabaseManager:
    """
    Multi-dialect database manager.

    Args:
        db_id:             Logical identifier for the database.
        connection_string: SQLAlchemy URL. Defaults to in-memory SQLite.
        dialect:           Dialect hint for LLM prompts ("sqlite", "postgresql",
                           "mssql").
    """

    def __init__(
        self,
        db_id: str = "default",
        connection_string: Optional[str] = None,
        dialect: str = "sqlite",
    ) -> None:
        self.db_id = db_id
        self.dialect = dialect
        self._lock = threading.Lock()

        if connection_string is None:
            connection_string = "sqlite:///:memory:"

        engine_kwargs: dict = {}
        if "sqlite" in connection_string:
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        if connection_string == "sqlite:///:memory:":
            engine_kwargs["poolclass"] = StaticPool

        self._engine = sa.create_engine(connection_string, **engine_kwargs)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def get_schema(self) -> DatabaseSchema:
        inspector = sa_inspect(self._engine)
        tables = []
        for table_name in inspector.get_table_names():
            cols = [
                ColumnInfo(name=c["name"], type=str(c["type"]))
                for c in inspector.get_columns(table_name)
            ]
            tables.append(TableSchema(name=table_name, columns=cols))
        return DatabaseSchema(tables=tables)

    def schema_prompt_context(self) -> str:
        """Return a human-readable schema string for LLM prompts."""
        schema = self.get_schema()
        lines = [f"Database dialect: {self.dialect.upper()}"]
        for table in schema.tables:
            lines.append(f"\nTable: {table.name}")
            for col in table.columns:
                lines.append(f"  - {col.name} ({col.type})")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Sample data
    # ------------------------------------------------------------------

    def load_sample_data(self) -> None:
        """Create and populate sample tables (idempotent)."""
        with self._lock:
            with self._engine.begin() as conn:
                for ddl in _SAMPLE_DDL:
                    conn.execute(text(ddl))
                for table, (stmt, rows) in _SAMPLE_INSERTS.items():
                    count = conn.execute(
                        text(f"SELECT COUNT(*) FROM {table}")
                    ).scalar()
                    if count == 0:
                        conn.execute(text(stmt), rows)

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def execute(self, sql: str) -> ExecutionResult:
        """
        Execute *sql* and return an :class:`ExecutionResult`.

        Write/DDL keywords are blocked and return an error result without
        touching the database.
        """
        sql = sql.strip()
        if not sql:
            return ExecutionResult(success=False, error="Empty query.")

        if not _is_safe_sql(sql):
            return ExecutionResult(
                success=False,
                error=(
                    "Blocked: query contains a disallowed write/DDL keyword. "
                    "Only SELECT queries are permitted."
                ),
            )

        try:
            with self._lock:
                with self._engine.connect() as conn:
                    result = conn.execute(text(sql))
                    columns = list(result.keys())
                    rows = [tuple(r) for r in result.fetchmany(MAX_ROWS)]
                    return ExecutionResult(
                        success=True,
                        rows=rows,
                        columns=columns,
                        row_count=len(rows),
                    )
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._engine.dispose()


# ---------------------------------------------------------------------------
# DBManager  (legacy – kept for tasks.py compatibility)
# ---------------------------------------------------------------------------

class DBManager:
    """Thin SQLite wrapper kept for backward compatibility with tasks.py."""

    def __init__(self, db_path: str = "app.db") -> None:
        self.db_path = db_path
        conn_str = f"sqlite:///{db_path}"
        self._manager = DatabaseManager(
            db_id=db_path,
            connection_string=conn_str,
        )
        with self._manager._engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS users "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age INTEGER, email TEXT)"
            ))
            count = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
            if count == 0:
                conn.execute(text(
                    "INSERT INTO users (name, age, email) VALUES (:n, :a, :e)"
                ), [
                    {"n": "Alice",   "a": 28, "e": "alice@example.com"},
                    {"n": "Bob",     "a": 32, "e": "bob@example.com"},
                    {"n": "Charlie", "a": 24, "e": "charlie@example.com"},
                ])

    def execute_query(self, query: str):
        result = self._manager.execute(query)
        if not result.success:
            raise RuntimeError(result.error)
        return [dict(zip(result.columns, row)) for row in result.rows]

"""
Database manager: schema introspection and SQL execution.

Supports SQLite out of the box.  Additional backends (PostgreSQL, MySQL,
etc.) can be added by supplying a PEP 249-compliant connection object.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    name: str
    dtype: str
    nullable: bool = True
    primary_key: bool = False


@dataclass
class TableSchema:
    name: str
    columns: List[ColumnInfo] = field(default_factory=list)

    def to_ddl(self) -> str:
        """Return a compact DDL-style string for use in prompts."""
        cols = ", ".join(
            f"{c.name} {c.dtype}{'  PK' if c.primary_key else ''}"
            for c in self.columns
        )
        return f"CREATE TABLE {self.name} ({cols});"


@dataclass
class DatabaseSchema:
    db_id: str
    tables: List[TableSchema] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Format the schema for inclusion in an LLM prompt."""
        lines = [f"-- Database: {self.db_id}"]
        for table in self.tables:
            lines.append(table.to_ddl())
        return "\n".join(lines)


@dataclass
class ExecutionResult:
    success: bool
    rows: List[Tuple[Any, ...]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    error: Optional[str] = None
    row_count: int = 0

    def to_display(self, max_rows: int = 10) -> str:
        if not self.success:
            return f"ERROR: {self.error}"
        if not self.rows:
            return "(no rows returned)"
        header = " | ".join(self.columns)
        separator = "-" * len(header)
        rows_str = "\n".join(
            " | ".join(str(v) for v in row) for row in self.rows[:max_rows]
        )
        suffix = f"\n... ({self.row_count} rows total)" if self.row_count > max_rows else ""
        return f"{header}\n{separator}\n{rows_str}{suffix}"


class DatabaseManager:
    """
    Wraps a database connection and provides schema introspection and
    query execution.

    Args:
        db_path: Path to an SQLite database file, or ``:memory:`` for an
            in-memory SQLite database.
        db_id: Human-readable identifier for the database (used in prompts).
        connection: An existing PEP-249 connection.  If supplied, *db_path*
            is ignored.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        db_id: str = "default",
        connection=None,
    ) -> None:
        self.db_id = db_id
        if connection is not None:
            self._conn = connection
        else:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        self._schema: Optional[DatabaseSchema] = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def get_schema(self, refresh: bool = False) -> DatabaseSchema:
        """
        Introspect the database and return a :class:`DatabaseSchema`.

        The result is cached; pass ``refresh=True`` to force re-inspection.
        """
        if self._schema is not None and not refresh:
            return self._schema

        cursor = self._conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        table_names = [row[0] for row in cursor.fetchall()]

        tables: List[TableSchema] = []
        for tname in table_names:
            cursor.execute(f"PRAGMA table_info({tname});")
            cols = []
            for col in cursor.fetchall():
                cols.append(
                    ColumnInfo(
                        name=col[1],
                        dtype=col[2],
                        nullable=not bool(col[3]),
                        primary_key=bool(col[5]),
                    )
                )
            tables.append(TableSchema(name=tname, columns=cols))

        self._schema = DatabaseSchema(db_id=self.db_id, tables=tables)
        return self._schema

    def schema_prompt_context(self) -> str:
        """Return the schema formatted for inclusion in a prompt."""
        return self.get_schema().to_prompt_context()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, sql: str) -> ExecutionResult:
        """
        Execute *sql* and return an :class:`ExecutionResult`.

        Read-only queries (SELECT / WITH) return rows; write queries
        (INSERT / UPDATE / DELETE / CREATE / DROP) return row-count only.
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute(sql)
            sql_upper = sql.strip().upper()
            if sql_upper.startswith(("SELECT", "WITH", "PRAGMA")):
                rows = [tuple(row) for row in cursor.fetchall()]
                columns = (
                    [desc[0] for desc in cursor.description]
                    if cursor.description
                    else []
                )
                return ExecutionResult(
                    success=True,
                    rows=rows,
                    columns=columns,
                    row_count=len(rows),
                )
            else:
                self._conn.commit()
                return ExecutionResult(
                    success=True,
                    row_count=cursor.rowcount,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("SQL execution error: %s | SQL: %s", exc, sql)
            return ExecutionResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def load_sample_data(self) -> None:
        """
        Populate the in-memory DB with a small sample schema and data
        suitable for demonstrations and tests.
        """
        ddl = """
        CREATE TABLE IF NOT EXISTS employees (
            id       INTEGER PRIMARY KEY,
            name     TEXT    NOT NULL,
            dept     TEXT    NOT NULL,
            salary   REAL    NOT NULL,
            hire_date TEXT
        );
        CREATE TABLE IF NOT EXISTS departments (
            id      INTEGER PRIMARY KEY,
            name    TEXT    NOT NULL,
            budget  REAL
        );
        CREATE TABLE IF NOT EXISTS projects (
            id          INTEGER PRIMARY KEY,
            name        TEXT NOT NULL,
            dept        TEXT NOT NULL,
            start_date  TEXT,
            end_date    TEXT
        );
        """
        data = """
        INSERT OR IGNORE INTO departments VALUES (1,'Engineering',500000);
        INSERT OR IGNORE INTO departments VALUES (2,'Marketing',200000);
        INSERT OR IGNORE INTO departments VALUES (3,'HR',150000);

        INSERT OR IGNORE INTO employees VALUES (1,'Alice','Engineering',95000,'2020-01-15');
        INSERT OR IGNORE INTO employees VALUES (2,'Bob','Marketing',72000,'2019-06-01');
        INSERT OR IGNORE INTO employees VALUES (3,'Carol','Engineering',88000,'2021-03-10');
        INSERT OR IGNORE INTO employees VALUES (4,'David','HR',65000,'2018-09-20');
        INSERT OR IGNORE INTO employees VALUES (5,'Eve','Engineering',102000,'2017-11-05');
        INSERT OR IGNORE INTO employees VALUES (6,'Frank','Marketing',78000,'2022-02-28');

        INSERT OR IGNORE INTO projects VALUES (1,'Alpha','Engineering','2023-01-01','2023-12-31');
        INSERT OR IGNORE INTO projects VALUES (2,'Beta','Marketing','2023-03-01',NULL);
        INSERT OR IGNORE INTO projects VALUES (3,'Gamma','Engineering','2022-06-01','2023-06-30');
        """
        cursor = self._conn.cursor()
        for stmt in ddl.split(";"):
            stmt = stmt.strip()
            if stmt:
                cursor.execute(stmt)
        for stmt in data.split(";"):
            stmt = stmt.strip()
            if stmt:
                cursor.execute(stmt)
        self._conn.commit()
        self._schema = None  # invalidate cached schema

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

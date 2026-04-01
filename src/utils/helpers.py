"""
Shared helper utilities.
"""

from __future__ import annotations

import re
from typing import List

import sqlparse


def extract_sql(text: str) -> str:
    """
    Extract the first SQL statement from *text*.

    Handles LLM outputs that wrap the SQL in markdown code fences
    (e.g., ```sql ... ```) or that include explanatory prose before
    the query.
    """
    # Try code-fence extraction first (```sql ... ``` or ``` ... ```)
    fence_pattern = re.compile(
        r"```(?:sql)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE
    )
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()

    # Fall back: find the first SELECT / WITH / INSERT / UPDATE / DELETE statement
    sql_start = re.compile(
        r"\b(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b",
        re.IGNORECASE,
    )
    m = sql_start.search(text)
    if m:
        return text[m.start():].strip()

    return text.strip()


def format_sql(sql: str) -> str:
    """Return *sql* formatted for readability."""
    try:
        return sqlparse.format(sql, reindent=True, keyword_case="upper").strip()
    except Exception:  # noqa: BLE001
        return sql.strip()


def build_few_shot_block(examples: list) -> str:
    """
    Build a few-shot prompt block from a list of :class:`Experience` objects.

    Returns an empty string when *examples* is empty.
    """
    if not examples:
        return ""

    lines: List[str] = ["-- Few-shot examples from memory:"]
    for i, ex in enumerate(examples, start=1):
        # Use corrected SQL if available, otherwise the stored SQL
        sql_to_show = ex.correction if ex.correction else ex.sql
        lines.append(f"-- Example {i}")
        lines.append(f"-- Q: {ex.question}")
        lines.append(f"-- A: {sql_to_show}")
    lines.append("")  # blank line separator
    return "\n".join(lines)

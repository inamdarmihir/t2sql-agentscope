"""
Tests for the Text-to-SQL + Qdrant RL self-improvement system.

These tests are designed to run without an LLM or external Qdrant server.
They use:
  - An in-memory Qdrant instance (the default when no URL is provided).
  - A simple hash-based embedding function (no model download).
  - Stub AgentScope agents (no credentials required).
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
import unittest

# ---------------------------------------------------------------------------
# Minimal agentscope stubs (allows tests to run without the package)
# ---------------------------------------------------------------------------

def _install_agentscope_stubs():
    if "agentscope" in sys.modules:
        return  # already available (real or stub)

    agentscope = types.ModuleType("agentscope")
    agentscope.init = lambda **kw: None  # type: ignore
    sys.modules["agentscope"] = agentscope

    agents_mod = types.ModuleType("agentscope.agents")
    sys.modules["agentscope.agents"] = agents_mod

    class _AgentBase:
        def __init__(self, name="agent", model_config_name="stub", **kwargs):
            self.name = name
            self.model_config_name = model_config_name

        def __call__(self, msg):
            return self.reply(msg)

        def reply(self, x):
            raise NotImplementedError

    agents_mod.AgentBase = _AgentBase  # type: ignore

    msg_mod = types.ModuleType("agentscope.message")
    sys.modules["agentscope.message"] = msg_mod

    class _Msg:
        def __init__(self, name, content, role, **kwargs):
            self.name = name
            self.content = content
            self.role = role
            for k, v in kwargs.items():
                setattr(self, k, v)

    msg_mod.Msg = _Msg  # type: ignore


_install_agentscope_stubs()

# ---------------------------------------------------------------------------
# Project imports (after stubs)
# ---------------------------------------------------------------------------

from src.database.db_manager import DatabaseManager  # noqa: E402
from src.memory.qdrant_memory import Experience, QdrantMemory  # noqa: E402
from src.rl_loop import RLLoop  # noqa: E402
from src.utils.helpers import build_few_shot_block, extract_sql, format_sql  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _hash_embed(text: str, dim: int = 64) -> list:
    digest = hashlib.sha256(text.encode()).digest()
    vec = [(b / 255.0) * 2 - 1 for b in digest]
    while len(vec) < dim:
        vec.extend(vec)
    return vec[:dim]


def embed_fn(text: str) -> list:
    return _hash_embed(text, 64)


VECTOR_SIZE = 64


def make_memory() -> QdrantMemory:
    return QdrantMemory(embed_fn=embed_fn, vector_size=VECTOR_SIZE)


# ---------------------------------------------------------------------------
# Tests: Helpers
# ---------------------------------------------------------------------------

class TestExtractSQL(unittest.TestCase):
    def test_plain_sql(self):
        assert extract_sql("SELECT * FROM t") == "SELECT * FROM t"

    def test_markdown_fence(self):
        text = "Here is the query:\n```sql\nSELECT id FROM users;\n```"
        assert "SELECT" in extract_sql(text)

    def test_fence_no_language(self):
        text = "```\nSELECT 1\n```"
        result = extract_sql(text)
        assert "SELECT" in result

    def test_prose_before_sql(self):
        text = "Sure! The answer is: SELECT name FROM employees WHERE dept='HR'"
        assert "SELECT" in extract_sql(text)

    def test_empty(self):
        assert extract_sql("") == ""


class TestFormatSQL(unittest.TestCase):
    def test_basic(self):
        raw = "select id,name from employees where dept='HR'"
        formatted = format_sql(raw)
        assert "SELECT" in formatted or "select" in formatted.lower()


class TestBuildFewShotBlock(unittest.TestCase):
    def test_empty(self):
        assert build_few_shot_block([]) == ""

    def test_with_examples(self):
        exp = Experience(
            question="Count employees",
            sql="SELECT COUNT(*) FROM employees",
            schema_context="",
            reward=1.0,
        )
        block = build_few_shot_block([exp])
        assert "Count employees" in block
        assert "SELECT COUNT" in block

    def test_correction_preferred(self):
        exp = Experience(
            question="Q",
            sql="WRONG SQL",
            schema_context="",
            reward=0.2,
            correction="SELECT 1",
        )
        block = build_few_shot_block([exp])
        assert "SELECT 1" in block
        assert "WRONG SQL" not in block


# ---------------------------------------------------------------------------
# Tests: DatabaseManager
# ---------------------------------------------------------------------------

class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        self.db = DatabaseManager(db_id="test")
        self.db.load_sample_data()

    def tearDown(self):
        self.db.close()

    def test_schema_introspection(self):
        schema = self.db.get_schema()
        table_names = [t.name for t in schema.tables]
        assert "employees" in table_names
        assert "departments" in table_names

    def test_schema_prompt_context(self):
        ctx = self.db.schema_prompt_context()
        assert "employees" in ctx
        assert "salary" in ctx

    def test_select_query(self):
        result = self.db.execute("SELECT COUNT(*) FROM employees")
        assert result.success is True
        assert result.row_count == 1
        assert result.rows[0][0] == 6

    def test_avg_salary(self):
        result = self.db.execute(
            "SELECT dept, AVG(salary) FROM employees GROUP BY dept ORDER BY dept"
        )
        assert result.success is True
        assert result.row_count > 0

    def test_syntax_error(self):
        result = self.db.execute("SELEC broken sql")
        assert result.success is False
        assert result.error is not None

    def test_no_rows(self):
        result = self.db.execute("SELECT * FROM employees WHERE id = 9999")
        assert result.success is True
        assert result.row_count == 0

    def test_display(self):
        result = self.db.execute("SELECT name FROM employees LIMIT 2")
        display = result.to_display()
        assert "name" in display.lower()


# ---------------------------------------------------------------------------
# Tests: QdrantMemory
# ---------------------------------------------------------------------------

class TestQdrantMemory(unittest.TestCase):
    def setUp(self):
        self.memory = make_memory()

    def test_store_and_retrieve(self):
        exp = Experience(
            question="How many employees are there?",
            sql="SELECT COUNT(*) FROM employees",
            schema_context="employees(id, name, dept, salary)",
            reward=1.0,
            db_id="test",
        )
        point_id = self.memory.store(exp)
        assert point_id is not None

        results = self.memory.retrieve(
            "How many employees are there?",
            top_k=3,
            db_id="test",
        )
        assert len(results) == 1
        assert results[0].sql == exp.sql

    def test_min_reward_filter(self):
        self.memory.store(
            Experience(
                question="Bad query",
                sql="SELECT WRONG",
                schema_context="",
                reward=0.1,
                db_id="test",
            )
        )
        results = self.memory.retrieve("Bad query", top_k=5, min_reward=0.5, db_id="test")
        assert len(results) == 0

    def test_collection_size(self):
        assert self.memory.collection_size() == 0
        self.memory.store(
            Experience(question="Q", sql="SELECT 1", schema_context="", reward=1.0)
        )
        assert self.memory.collection_size() == 1

    def test_clear(self):
        self.memory.store(
            Experience(question="Q", sql="SELECT 1", schema_context="", reward=1.0)
        )
        self.memory.clear()
        assert self.memory.collection_size() == 0

    def test_experience_payload_roundtrip(self):
        exp = Experience(
            question="Q",
            sql="SELECT 1",
            schema_context="ctx",
            reward=0.8,
            db_id="db1",
            correction="SELECT 2",
        )
        payload = exp.to_payload()
        recovered = Experience.from_payload(payload)
        assert recovered.question == exp.question
        assert recovered.sql == exp.sql
        assert recovered.reward == exp.reward
        assert recovered.correction == exp.correction

    def test_multiple_experiences_retrieval(self):
        for i in range(5):
            self.memory.store(
                Experience(
                    question=f"Question {i} about employees",
                    sql=f"SELECT {i} FROM employees",
                    schema_context="employees(id, name)",
                    reward=1.0,
                )
            )
        results = self.memory.retrieve("Question about employees", top_k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Tests: RL Loop (end-to-end, no LLM)
# ---------------------------------------------------------------------------

class TestRLLoop(unittest.TestCase):
    """End-to-end tests using stub agents and in-memory Qdrant + SQLite."""

    def _make_loop(self, sql_map=None):
        """Build a RLLoop with patched agents."""
        from src.agents.feedback_agent import FeedbackAgent
        from src.agents.t2sql_agent import T2SQLAgent

        memory = make_memory()
        db = DatabaseManager(db_id="sample")
        db.load_sample_data()

        t2sql = T2SQLAgent(
            name="T2SQL",
            model_config_name="stub",
            memory=memory,
            top_k_examples=2,
            db_id="sample",
        )
        feedback = FeedbackAgent(
            name="Feedback",
            model_config_name="stub",
            use_llm=False,
        )

        _sql_map = sql_map or {
            "count": "SELECT COUNT(*) AS cnt FROM employees",
        }

        import types as _types
        from agentscope.message import Msg  # type: ignore

        def _reply_t2sql(self_agent, x):
            raw_content = x.content if x else "{}"
            if isinstance(raw_content, str):
                try:
                    content = json.loads(raw_content)
                except Exception:
                    content = {"question": raw_content}
            else:
                content = raw_content
            q = content.get("question", "").lower()
            sql = "SELECT 1"
            for key, val in _sql_map.items():
                if key in q:
                    sql = val
                    break
            return Msg(name=self_agent.name, content=sql, role="assistant")

        t2sql.reply = _types.MethodType(_reply_t2sql, t2sql)

        loop = RLLoop(
            t2sql_agent=t2sql,
            feedback_agent=feedback,
            db_manager=db,
            qdrant_memory=memory,
            store_threshold=0.5,
            apply_corrections=True,
        )
        return loop, memory

    def test_correct_query_stored(self):
        loop, memory = self._make_loop({"count employees": "SELECT COUNT(*) FROM employees"})
        result = loop.run_once("count employees total")
        assert result.sql != ""
        assert result.execution_result.success
        assert memory.collection_size() >= 1

    def test_reward_positive_for_rows(self):
        loop, memory = self._make_loop(
            {"employees": "SELECT name, salary FROM employees"}
        )
        result = loop.run_once("list all employees")
        assert result.reward >= 0.5
        assert result.correct

    def test_reward_zero_for_error(self):
        loop, memory = self._make_loop({"broken": "INVALID SQL !!!"})
        result = loop.run_once("broken query")
        assert result.reward == 0.0
        assert not result.correct

    def test_memory_grows_over_iterations(self):
        loop, memory = self._make_loop(
            {
                "count": "SELECT COUNT(*) FROM employees",
                "avg": "SELECT AVG(salary) FROM employees",
            }
        )
        loop.run_once("count employees")
        loop.run_once("get avg salary")
        assert memory.collection_size() >= 2

    def test_batch_run(self):
        loop, memory = self._make_loop(
            {"count": "SELECT COUNT(*) FROM employees"}
        )
        results = loop.run_batch(["count employees", "count employees again"])
        assert len(results) == 2

    def test_loop_result_summary(self):
        loop, memory = self._make_loop({"count": "SELECT COUNT(*) FROM employees"})
        result = loop.run_once("count employees")
        summary = result.summary()
        assert "Question" in summary
        assert "SQL" in summary
        assert "Reward" in summary


if __name__ == "__main__":
    unittest.main()

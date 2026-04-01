"""
Runnable demo that exercises the full T2SQL + Qdrant RL pipeline
without requiring an external LLM or Qdrant server.

It monkey-patches the model call on the agents to return deterministic
SQL strings, then exercises the complete RL loop: generate → execute →
evaluate → store → retrieve.

Run:
    python demo.py
"""

from __future__ import annotations

import json
import logging
import sys
import types

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Stub agentscope so we can run the demo without credentials
# ---------------------------------------------------------------------------
try:
    import agentscope  # noqa: F401
except ImportError:
    agentscope = types.ModuleType("agentscope")
    agentscope.init = lambda **kw: None  # type: ignore[attr-defined]
    sys.modules["agentscope"] = agentscope
    agents_mod = types.ModuleType("agentscope.agents")
    sys.modules["agentscope.agents"] = agents_mod

    class _AgentBase:  # minimal stub
        def __init__(self, name="agent", model_config_name="stub", **kwargs):
            self.name = name
            self.model_config_name = model_config_name

        def __call__(self, msg):
            return self.reply(msg)

        def reply(self, x):
            raise NotImplementedError

    agents_mod.AgentBase = _AgentBase  # type: ignore[attr-defined]
    msg_mod = types.ModuleType("agentscope.message")
    sys.modules["agentscope.message"] = msg_mod

    class _Msg:  # minimal stub
        def __init__(self, name, content, role, **kwargs):
            self.name = name
            self.content = content
            self.role = role
            for k, v in kwargs.items():
                setattr(self, k, v)

    msg_mod.Msg = _Msg  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Now import project modules (they use the stubs above if agentscope is absent)
# ---------------------------------------------------------------------------
from src.agents.feedback_agent import FeedbackAgent  # noqa: E402
from src.agents.t2sql_agent import T2SQLAgent  # noqa: E402
from src.database.db_manager import DatabaseManager  # noqa: E402
from src.memory.qdrant_memory import QdrantMemory  # noqa: E402
from src.rl_loop import RLLoop  # noqa: E402

# ---------------------------------------------------------------------------
# Deterministic embedding function (no model download required)
# ---------------------------------------------------------------------------
import hashlib  # noqa: E402


def _demo_embed(text: str, dim: int = 64):
    """Simple hash-based pseudo-embedding for demo / test purposes."""
    digest = hashlib.sha256(text.encode()).digest()
    vec = [(b / 255.0) * 2 - 1 for b in digest]
    # Pad or truncate to `dim`
    while len(vec) < dim:
        vec.extend(vec)
    return vec[:dim]


DEMO_EMBED_DIM = 64


def demo_embed_fn(text: str):
    return _demo_embed(text, DEMO_EMBED_DIM)


# ---------------------------------------------------------------------------
# Deterministic SQL responses for demo purposes
# ---------------------------------------------------------------------------
_DEMO_SQL_MAP = {
    "average salary per department": "SELECT dept, AVG(salary) AS avg_salary FROM employees GROUP BY dept",
    "highest paid employee": "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1",
    "number of employees": "SELECT COUNT(*) AS total FROM employees",
    "employees in engineering": "SELECT name FROM employees WHERE dept = 'Engineering'",
    "total budget for all departments": "SELECT SUM(budget) AS total_budget FROM departments",
    "projects that have ended": "SELECT name FROM projects WHERE end_date IS NOT NULL",
}


def _make_sql_agent(agent: T2SQLAgent):
    """Patch the agent to return deterministic SQL without calling an LLM."""

    def _patched_reply(self, x):
        from agentscope.message import Msg  # type: ignore[attr-defined]

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
        for key, val in _DEMO_SQL_MAP.items():
            if key in q:
                sql = val
                break
        return Msg(name=self.name, content=sql, role="assistant")

    import types

    agent.reply = types.MethodType(_patched_reply, agent)
    return agent


def _make_feedback_agent(agent: FeedbackAgent):
    """Patch the feedback agent to skip LLM calls."""
    agent.use_llm = False
    return agent


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def run_demo():
    print("\n" + "=" * 70)
    print("  Text-to-SQL + AgentScope + Qdrant RL Self-Improvement  Demo")
    print("=" * 70 + "\n")

    # Set up components
    memory = QdrantMemory(embed_fn=demo_embed_fn, vector_size=DEMO_EMBED_DIM)
    db = DatabaseManager(db_id="sample")
    db.load_sample_data()

    t2sql = T2SQLAgent(
        name="T2SQLAgent",
        model_config_name="stub",
        memory=memory,
        top_k_examples=2,
        db_id="sample",
    )
    feedback = FeedbackAgent(
        name="FeedbackAgent",
        model_config_name="stub",
        use_llm=False,
    )

    _make_sql_agent(t2sql)
    _make_feedback_agent(feedback)

    loop = RLLoop(
        t2sql_agent=t2sql,
        feedback_agent=feedback,
        db_manager=db,
        qdrant_memory=memory,
        store_threshold=0.5,
        apply_corrections=True,
    )

    questions = [
        "What is the average salary per department?",
        "Who is the highest paid employee?",
        "How many employees are there in total?",
        "List all employees in engineering",
        "What is the total budget for all departments?",
        "Which projects have ended?",
        # Re-run the first question to demonstrate that the agent now
        # retrieves it from memory as a few-shot example
        "What is the average salary per department?",
    ]

    results = []
    for i, question in enumerate(questions, start=1):
        print(f"\n--- Iteration {i} ---")
        result = loop.run_once(question)
        results.append(result)
        print(result.summary())
        print(f"[Memory size: {memory.collection_size()}]")

    print("\n" + "=" * 70)
    correct = sum(1 for r in results if r.correct)
    print(f"Summary: {correct}/{len(results)} queries correct")
    print(f"Final Qdrant memory size: {memory.collection_size()} experience(s)")
    print("=" * 70 + "\n")
    return results


if __name__ == "__main__":
    run_demo()

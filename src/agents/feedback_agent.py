"""
Feedback / execution agent.

Supports two calling modes:

1. **RL-loop mode** – ``x.content`` is a JSON string produced by
   :class:`~src.rl_loop.RLLoop._get_feedback` containing
   ``question``, ``sql``, ``schema_context``, and ``execution_result``.
   The agent scores the already-executed result and returns a JSON reply
   with ``score``, ``correct``, ``reasoning``, and optionally
   ``corrected_sql``.

2. **Pipeline mode** (tasks.py) – ``x.content`` is a raw SQL string
   (possibly wrapped in markdown fences).  The agent executes it against
   the database and returns a plain-text result message.

When ``use_llm=False`` (default for tests/demo) scoring is deterministic.
When ``use_llm=True`` the LLM is called for richer reasoning.
"""

from __future__ import annotations

import json
from typing import Optional

from agentscope.agents import AgentBase
from agentscope.message import Msg

from src.database.db_manager import DBManager
from src.utils.helpers import extract_sql


class FeedbackAgent(AgentBase):
    """
    Args:
        name:               AgentScope agent name.
        model_config_name:  Model config key.
        use_llm:            When *False* scoring is fully deterministic (good
                            for tests and demo).  When *True* the LLM scores
                            the execution result.
        sys_prompt:         Optional custom system prompt.
    """

    def __init__(
        self,
        name: str = "FeedbackAgent",
        model_config_name: str = "ollama_chat_config",
        use_llm: bool = True,
        sys_prompt: str = (
            "You are a SQL execution evaluator. "
            "Given a question, SQL, and its execution result, respond with a "
            "JSON object containing:\n"
            "  score      (float 0.0-1.0)\n"
            "  correct    (bool)\n"
            "  reasoning  (string)\n"
            "  corrected_sql (string or null)\n"
            "Output ONLY valid JSON."
        ),
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            **kwargs,
        )
        self.use_llm = use_llm
        self._db = DBManager()

    # ------------------------------------------------------------------

    def reply(self, x: Optional[Msg] = None) -> Msg:
        content = x.content if x else ""

        # ---- RL-loop mode: content is JSON with execution_result ----
        if isinstance(content, str) and content.strip().startswith("{"):
            try:
                data = json.loads(content)
            except (ValueError, TypeError):
                data = None

            if data and "execution_result" in data:
                return self._score_result(data)

        # ---- Pipeline mode: content is SQL (possibly in markdown) ----
        sql = extract_sql(content) if isinstance(content, str) else str(content)
        return self._execute_and_report(sql)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_result(self, data: dict) -> Msg:
        """Score an already-executed result (RL-loop mode)."""
        exec_result = data.get("execution_result", {})
        success = exec_result.get("success", False)
        rows = exec_result.get("rows", [])
        error = exec_result.get("error")

        if self.use_llm:
            # Delegate scoring to the LLM
            prompt = Msg(
                name="user",
                content=json.dumps(data),
                role="user",
            )
            llm_response = super().reply(prompt)
            try:
                parsed = json.loads(llm_response.content or "{}")
                return Msg(name=self.name, content=json.dumps(parsed), role="assistant")
            except (ValueError, TypeError):
                pass  # fall through to deterministic scoring

        # Deterministic scoring
        if not success:
            result = {"score": 0.0, "correct": False,
                      "reasoning": f"Execution error: {error}", "corrected_sql": None}
        elif rows:
            result = {"score": 1.0, "correct": True,
                      "reasoning": f"Query returned {len(rows)} row(s).", "corrected_sql": None}
        else:
            result = {"score": 0.7, "correct": True,
                      "reasoning": "Query succeeded but returned no rows.", "corrected_sql": None}

        return Msg(name=self.name, content=json.dumps(result), role="assistant")

    def _execute_and_report(self, sql: str) -> Msg:
        """Execute raw SQL and return a plain-text result (pipeline mode)."""
        try:
            rows = self._db.execute_query(sql)
            feedback = f"Execution successful. Result: {rows}"
        except Exception as exc:  # noqa: BLE001
            feedback = f"Execution failed. Error: {exc}"

        return Msg(name=self.name, content=feedback, role="assistant")

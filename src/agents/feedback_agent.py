"""
Feedback / evaluation agent for the RL self-improvement loop.

This agent:
1. Receives a (question, sql, execution_result) tuple.
2. Optionally invokes an LLM for semantic correctness checking.
3. Returns a scalar reward in [0, 1] and an optional SQL correction.

Two modes are supported:
- **Execution-only** (``use_llm=False``): reward = 1.0 if the SQL executed
  successfully and returned rows, 0.0 otherwise.
- **LLM-assisted** (``use_llm=True``): after execution-based scoring the
  agent also asks the LLM whether the answer is semantically correct and
  may provide a corrected query.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional, Union

from agentscope.agents import AgentBase
from agentscope.message import Msg

from src.utils.helpers import extract_sql, format_sql

logger = logging.getLogger(__name__)

_FEEDBACK_SYSTEM_PROMPT = """\
You are a SQL correctness evaluator.  Given:
  - A natural-language question
  - A database schema
  - A SQL query
  - The query's execution result (or error message)

Your task:
1. Decide whether the SQL correctly answers the question (score 1.0) or not (score 0.0).
   A score between 0 and 1 is allowed for partial credit.
2. If the SQL is wrong or partially wrong, provide a corrected SQL query.

Respond ONLY with a JSON object like:
{
  "score": 0.9,
  "correct": true,
  "reasoning": "one sentence",
  "corrected_sql": null
}
Set "corrected_sql" to null if the original SQL was correct.
"""


class FeedbackAgent(AgentBase):
    """
    Evaluates the quality of a generated SQL query and provides a reward.

    Args:
        name: Agent name.
        model_config_name: AgentScope model config name.
        use_llm: Whether to use the LLM for semantic correctness checking.
            If *False*, only execution-based scoring is used (faster, cheaper).
    """

    def __init__(
        self,
        name: str = "FeedbackAgent",
        model_config_name: str = "gpt-4o",
        use_llm: bool = True,
    ) -> None:
        super().__init__(name=name, model_config_name=model_config_name)
        self.use_llm = use_llm

    # ------------------------------------------------------------------
    # AgentBase interface
    # ------------------------------------------------------------------

    def reply(self, x: Union[Msg, None] = None) -> Msg:
        """
        Evaluate the SQL in *x* and return a feedback message.

        Expected *x.content* format (JSON string or dict):
        ``{"question": "...", "schema_context": "...", "sql": "...",
           "execution_result": {"success": bool, "rows": [...], "error": null}}``

        The returned ``Msg.content`` is a JSON string:
        ``{"score": float, "correct": bool, "reasoning": str, "corrected_sql": str|null}``
        """
        if x is None:
            return self._make_feedback(0.0, False, "No input provided.", None)

        raw_content = x.content
        if isinstance(raw_content, str):
            try:
                content = json.loads(raw_content)
            except json.JSONDecodeError:
                return self._make_feedback(0.0, False, "Invalid input format.", None)
        else:
            content = raw_content

        question: str = content.get("question", "")
        schema_context: str = content.get("schema_context", "")
        sql: str = content.get("sql", "")
        exec_result: dict = content.get("execution_result", {})

        score, reasoning, corrected_sql = self._evaluate(
            question, schema_context, sql, exec_result
        )
        correct = score >= 0.5

        return self._make_feedback(score, correct, reasoning, corrected_sql)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        question: str,
        schema_context: str,
        sql: str,
        exec_result: dict,
    ):
        """
        Compute a reward score.

        Returns:
            Tuple of (score: float, reasoning: str, corrected_sql: str | None)
        """
        execution_success = exec_result.get("success", False)
        execution_error = exec_result.get("error")
        rows = exec_result.get("rows", [])

        # Base score purely from execution
        if not execution_success:
            base_score = 0.0
            base_reasoning = f"SQL execution failed: {execution_error}"
        elif not rows:
            base_score = 0.5
            base_reasoning = "SQL executed successfully but returned no rows."
        else:
            base_score = 1.0
            base_reasoning = f"SQL executed successfully and returned {len(rows)} row(s)."

        if not self.use_llm or not question:
            return base_score, base_reasoning, None

        # Optionally refine with LLM evaluation
        return self._llm_evaluate(
            question, schema_context, sql, exec_result, base_score, base_reasoning
        )

    def _llm_evaluate(
        self,
        question: str,
        schema_context: str,
        sql: str,
        exec_result: dict,
        base_score: float,
        base_reasoning: str,
    ):
        """Use the LLM to semantically evaluate and optionally correct the SQL."""
        exec_summary = (
            f"Error: {exec_result.get('error')}"
            if not exec_result.get("success")
            else f"Rows returned: {exec_result.get('rows', [])[:5]}"
        )

        user_message = (
            f"Question: {question}\n\n"
            f"Schema:\n{schema_context}\n\n"
            f"SQL:\n{sql}\n\n"
            f"Execution result:\n{exec_summary}"
        )

        msgs = [
            Msg(name="system", content=_FEEDBACK_SYSTEM_PROMPT, role="system"),
            Msg(name="user", content=user_message, role="user"),
        ]

        try:
            response = self.model(msgs)
            raw = response.text if hasattr(response, "text") else str(response)
            # Strip optional code fences (```json ... ``` or ``` ... ```)
            fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
            clean = fence_match.group(1).strip() if fence_match else raw.strip()
            parsed = json.loads(clean)

            score = float(parsed.get("score", base_score))
            reasoning = parsed.get("reasoning", base_reasoning)
            corrected_sql_raw = parsed.get("corrected_sql")
            corrected_sql = (
                format_sql(extract_sql(corrected_sql_raw))
                if corrected_sql_raw
                else None
            )
            return score, reasoning, corrected_sql

        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM feedback parsing failed: %s. Falling back to base score.", exc)
            return base_score, base_reasoning, None

    # ------------------------------------------------------------------
    # Message factory
    # ------------------------------------------------------------------

    def _make_feedback(
        self,
        score: float,
        correct: bool,
        reasoning: str,
        corrected_sql: Optional[str],
    ) -> Msg:
        content = json.dumps(
            {
                "score": score,
                "correct": correct,
                "reasoning": reasoning,
                "corrected_sql": corrected_sql,
            }
        )
        return Msg(name=self.name, content=content, role="assistant")

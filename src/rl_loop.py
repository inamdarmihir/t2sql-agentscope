"""
RL self-improvement loop for Text-to-SQL.

The loop implements a simple episodic-memory reinforcement learning strategy:

    Loop:
        1. Receive a natural-language question.
        2. T2SQLAgent retrieves top-k similar past successes from Qdrant
           and uses them as few-shot demonstrations.
        3. T2SQLAgent generates SQL using the LLM.
        4. SQL is executed against the database.
        5. FeedbackAgent scores the (question, SQL, result) tuple.
        6. The experience (question, SQL, reward, correction) is stored in
           Qdrant for future retrieval.
        7. (Optional) If reward < threshold and a correction is available,
           the corrected SQL is executed and stored with a higher reward.

As Qdrant accumulates more positive experiences, retrieval quality improves
and the agent naturally generates better SQL over time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from agentscope.message import Msg

from src.agents.feedback_agent import FeedbackAgent
from src.agents.t2sql_agent import T2SQLAgent
from src.database.db_manager import DatabaseManager, ExecutionResult
from src.memory.qdrant_memory import Experience, QdrantMemory

logger = logging.getLogger(__name__)


@dataclass
class LoopResult:
    """Outcome of a single RL loop iteration."""

    question: str
    sql: str
    execution_result: ExecutionResult
    reward: float
    correct: bool
    reasoning: str
    corrected_sql: Optional[str] = None
    correction_result: Optional[ExecutionResult] = None
    memory_id: Optional[str] = None
    correction_memory_id: Optional[str] = None

    def summary(self) -> str:
        lines = [
            f"Question : {self.question}",
            f"SQL      : {self.sql}",
            f"Reward   : {self.reward:.2f}  ({'correct' if self.correct else 'incorrect'})",
            f"Reasoning: {self.reasoning}",
        ]
        if self.corrected_sql:
            lines.append(f"Correction: {self.corrected_sql}")
        lines.append(f"Result   : {self.execution_result.to_display()}")
        return "\n".join(lines)


class RLLoop:
    """
    Orchestrates the Text-to-SQL RL self-improvement loop.

    Args:
        t2sql_agent: The :class:`T2SQLAgent` that generates SQL.
        feedback_agent: The :class:`FeedbackAgent` that evaluates SQL.
        db_manager: The :class:`DatabaseManager` used to execute SQL.
        qdrant_memory: The :class:`QdrantMemory` used to store/retrieve
            experiences.
        store_threshold: Minimum reward for an experience to be stored as
            a positive example (default 0.5).
        apply_corrections: When *True* (default), automatically execute and
            store corrected SQL provided by the feedback agent.
    """

    def __init__(
        self,
        t2sql_agent: T2SQLAgent,
        feedback_agent: FeedbackAgent,
        db_manager: DatabaseManager,
        qdrant_memory: QdrantMemory,
        store_threshold: float = 0.5,
        apply_corrections: bool = True,
    ) -> None:
        self.t2sql_agent = t2sql_agent
        self.feedback_agent = feedback_agent
        self.db_manager = db_manager
        self.qdrant_memory = qdrant_memory
        self.store_threshold = store_threshold
        self.apply_corrections = apply_corrections

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def run_once(self, question: str) -> LoopResult:
        """
        Execute one full RL loop iteration for *question*.

        Returns:
            A :class:`LoopResult` containing the generated SQL, reward, and
            any stored memory IDs.
        """
        schema_context = self.db_manager.schema_prompt_context()
        db_id = self.db_manager.db_id

        # Step 1: Generate SQL
        sql = self._generate_sql(question, schema_context, db_id)

        # Step 2: Execute SQL
        exec_result = self.db_manager.execute(sql)

        # Step 3: Get feedback
        score, correct, reasoning, corrected_sql = self._get_feedback(
            question, schema_context, sql, exec_result
        )

        # Step 4: Store experience
        memory_id = self._store_experience(
            question=question,
            sql=sql,
            schema_context=schema_context,
            reward=score,
            db_id=db_id,
            correction=corrected_sql,
        )

        result = LoopResult(
            question=question,
            sql=sql,
            execution_result=exec_result,
            reward=score,
            correct=correct,
            reasoning=reasoning,
            corrected_sql=corrected_sql,
            memory_id=memory_id,
        )

        # Step 5: Apply correction if available and requested
        if self.apply_corrections and corrected_sql and not correct:
            correction_result = self.db_manager.execute(corrected_sql)
            if correction_result.success:
                correction_memory_id = self._store_experience(
                    question=question,
                    sql=corrected_sql,
                    schema_context=schema_context,
                    reward=1.0,
                    db_id=db_id,
                )
                result.correction_result = correction_result
                result.correction_memory_id = correction_memory_id
                logger.info("Stored corrected SQL with reward=1.0")

        logger.info(
            "RL iteration done | reward=%.2f | memory_size=%d",
            score,
            self.qdrant_memory.collection_size(),
        )
        return result

    def run_batch(self, questions: List[str]) -> List[LoopResult]:
        """
        Run :meth:`run_once` for each question in *questions*.

        Returns:
            A list of :class:`LoopResult` objects in the same order as
            *questions*.
        """
        results: List[LoopResult] = []
        for i, q in enumerate(questions, start=1):
            logger.info("Processing question %d/%d: %s", i, len(questions), q)
            results.append(self.run_once(q))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_sql(self, question: str, schema_context: str, db_id: str) -> str:
        """Ask the T2SQLAgent to generate SQL."""
        msg = Msg(
            name="user",
            content=json.dumps(
                {"question": question, "schema_context": schema_context, "db_id": db_id}
            ),
            role="user",
        )
        response = self.t2sql_agent(msg)
        return response.content or ""

    def _get_feedback(
        self, question: str, schema_context: str, sql: str, exec_result: ExecutionResult
    ):
        """Ask the FeedbackAgent to evaluate the SQL."""
        exec_dict = {
            "success": exec_result.success,
            "rows": exec_result.rows[:20],  # truncate for the prompt
            "columns": exec_result.columns,
            "error": exec_result.error,
        }
        msg = Msg(
            name="user",
            content=json.dumps(
                {
                    "question": question,
                    "schema_context": schema_context,
                    "sql": sql,
                    "execution_result": exec_dict,
                }
            ),
            role="user",
        )
        response = self.feedback_agent(msg)
        try:
            parsed = json.loads(response.content)
            return (
                float(parsed.get("score", 0.0)),
                bool(parsed.get("correct", False)),
                parsed.get("reasoning", ""),
                parsed.get("corrected_sql"),
            )
        except Exception:  # noqa: BLE001
            return 0.0, False, "Failed to parse feedback.", None

    def _store_experience(
        self,
        question: str,
        sql: str,
        schema_context: str,
        reward: float,
        db_id: str,
        correction: Optional[str] = None,
    ) -> Optional[str]:
        """Store an experience in Qdrant; return point ID or *None* on error."""
        try:
            exp = Experience(
                question=question,
                sql=sql,
                schema_context=schema_context,
                reward=reward,
                db_id=db_id,
                correction=correction,
            )
            return self.qdrant_memory.store(exp)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to store experience: %s", exc)
            return None

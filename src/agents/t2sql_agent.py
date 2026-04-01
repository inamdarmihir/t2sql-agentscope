"""
Text-to-SQL AgentScope agent.

This agent:
1. Accepts a natural-language question (and optional schema context).
2. Retrieves similar successful examples from :class:`QdrantMemory`.
3. Builds a few-shot prompt and calls the LLM to generate SQL.
4. Returns the generated SQL as an AgentScope ``Msg``.

The agent integrates with the RL self-improvement loop: every time it is
invoked, it augments its prompt with the best available in-memory examples,
so its performance naturally improves as the memory fills up.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Union

from agentscope.agents import AgentBase
from agentscope.message import Msg

from src.memory.qdrant_memory import QdrantMemory
from src.utils.helpers import build_few_shot_block, extract_sql, format_sql

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert SQL assistant.  Given a database schema and a natural-language question, \
produce a single, correct SQL query that answers the question.

Rules:
- Output ONLY the SQL statement, nothing else.
- Do NOT include any explanation, preamble, or trailing text.
- Use standard SQL syntax compatible with SQLite unless told otherwise.
- Qualify column names with table names when there is ambiguity.
- Use aliases to keep the query readable.
"""


class T2SQLAgent(AgentBase):
    """
    AgentScope agent that converts natural-language questions to SQL.

    Args:
        name: Agent name (shown in AgentScope logs).
        model_config_name: Name of the AgentScope model configuration to use.
        memory: A :class:`~src.memory.qdrant_memory.QdrantMemory` instance used
            to retrieve few-shot examples from past experiences.
        top_k_examples: How many similar examples to retrieve per query.
        db_id: Default database identifier used when filtering retrieved examples.
    """

    def __init__(
        self,
        name: str = "T2SQLAgent",
        model_config_name: str = "gpt-4o",
        memory: Optional[QdrantMemory] = None,
        top_k_examples: int = 3,
        db_id: str = "default",
    ) -> None:
        super().__init__(name=name, model_config_name=model_config_name)
        self.qdrant_memory = memory
        self.top_k_examples = top_k_examples
        self.db_id = db_id

    # ------------------------------------------------------------------
    # AgentBase interface
    # ------------------------------------------------------------------

    def reply(self, x: Union[Msg, None] = None) -> Msg:
        """
        Process an incoming message and return a SQL query.

        Expected *x.content* format (JSON string or dict):
        ``{"question": "...", "schema_context": "..."}``

        The ``schema_context`` key is optional; when absent the agent will
        generate SQL without schema grounding (not recommended).
        """
        if x is None:
            return Msg(name=self.name, content="", role="assistant")

        # Parse content
        raw_content = x.content
        if isinstance(raw_content, str):
            try:
                content = json.loads(raw_content)
            except json.JSONDecodeError:
                # Treat the whole string as a plain question
                content = {"question": raw_content}
        else:
            content = raw_content

        question: str = content.get("question", "")
        schema_context: str = content.get("schema_context", "")
        db_id: str = content.get("db_id", self.db_id)

        if not question:
            logger.warning("T2SQLAgent received a message with no question.")
            return Msg(name=self.name, content="", role="assistant")

        sql = self._generate_sql(question, schema_context, db_id)
        logger.info("Generated SQL: %s", sql)

        return Msg(
            name=self.name,
            content=sql,
            role="assistant",
            metadata={"question": question, "db_id": db_id},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_sql(
        self, question: str, schema_context: str, db_id: str
    ) -> str:
        """Retrieve examples, build the prompt, call the LLM, return SQL."""
        few_shot_block = ""
        if self.qdrant_memory is not None:
            examples = self.qdrant_memory.retrieve(
                question=question,
                schema_context=schema_context,
                top_k=self.top_k_examples,
                db_id=db_id,
            )
            few_shot_block = build_few_shot_block(examples)

        user_message = self._build_user_message(question, schema_context, few_shot_block)

        msgs = [
            Msg(name="system", content=_SYSTEM_PROMPT, role="system"),
            Msg(name="user", content=user_message, role="user"),
        ]

        response = self.model(msgs)
        raw_sql = response.text if hasattr(response, "text") else str(response)
        return format_sql(extract_sql(raw_sql))

    @staticmethod
    def _build_user_message(
        question: str, schema_context: str, few_shot_block: str
    ) -> str:
        parts: list[str] = []
        if schema_context:
            parts.append(f"Database schema:\n{schema_context}\n")
        if few_shot_block:
            parts.append(few_shot_block)
        parts.append(f"Question: {question}")
        parts.append("SQL:")
        return "\n".join(parts)

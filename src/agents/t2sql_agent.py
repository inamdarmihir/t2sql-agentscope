"""Text-to-SQL generation agent with Qdrant few-shot memory retrieval."""

from __future__ import annotations

import json
from typing import Optional

from agentscope.agents import AgentBase
from agentscope.message import Msg

from src.memory.qdrant_memory import QdrantMemory
from src.utils.helpers import build_few_shot_block, extract_sql


class T2SQLAgent(AgentBase):
    """
    Generates SQL from a natural-language question.

    Accepts two calling modes:

    1. **RL-loop mode** – ``x.content`` is a JSON string with keys
       ``question``, ``schema_context``, ``db_id`` (produced by
       :class:`~src.rl_loop.RLLoop`).

    2. **Pipeline mode** – ``x.content`` is a plain string (question or
       "User Query: … Schema Context: …" block from tasks.py).

    Args:
        name:               AgentScope agent name.
        model_config_name:  Model config key (used only when use_llm=True).
        memory:             :class:`QdrantMemory` instance.  When *None* the
                            agent skips few-shot context retrieval.
        top_k_examples:     Number of similar past queries to inject.
        db_id:              Database identifier used for memory filtering.
        sys_prompt:         Custom system prompt (optional).
    """

    def __init__(
        self,
        name: str = "T2SQLAgent",
        model_config_name: str = "ollama_chat_config",
        memory: Optional[QdrantMemory] = None,
        top_k_examples: int = 3,
        db_id: str = "",
        sys_prompt: str = (
            "You are an expert SQL developer. "
            "Convert the natural language question to a valid SQL query. "
            "Output ONLY the SQL query, nothing else."
        ),
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            **kwargs,
        )
        self._memory = memory
        self._top_k = top_k_examples
        self._db_id = db_id

    # ------------------------------------------------------------------

    def reply(self, x: Optional[Msg] = None) -> Msg:
        raw = x.content if x else ""

        # Parse structured input from RL loop
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                parsed = {"question": raw}
        else:
            parsed = raw if isinstance(raw, dict) else {"question": str(raw)}

        question = parsed.get("question", "")
        schema_context = parsed.get("schema_context", "")
        db_id = parsed.get("db_id", self._db_id)

        # Build few-shot prefix from memory (no mutation of x)
        few_shot = ""
        if self._memory and question:
            examples = self._memory.retrieve(question, top_k=self._top_k, db_id=db_id)
            few_shot = build_few_shot_block(examples)

        # Compose augmented prompt in a new Msg (never mutate x)
        parts = []
        if few_shot:
            parts.append(few_shot)
        if schema_context:
            parts.append(f"Schema:\n{schema_context}")
        parts.append(f"Question: {question}")

        augmented = Msg(
            name=x.name if x else "user",
            content="\n\n".join(parts),
            role=x.role if x else "user",
        )

        response = super().reply(augmented)

        # Extract clean SQL from any markdown fencing
        clean_sql = extract_sql(response.content or "")
        return Msg(name=self.name, content=clean_sql, role="assistant")

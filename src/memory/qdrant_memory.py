"""
Qdrant-based episodic memory for the RL self-improvement loop.

Each experience stored in Qdrant represents a (question, SQL, score) tuple.
During generation, the top-k most-similar successful examples are retrieved
and used as few-shot demonstrations, implementing an episodic-memory style
RL self-improvement strategy.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

logger = logging.getLogger(__name__)

# Collection names
EXPERIENCES_COLLECTION = "t2sql_experiences"

# Minimum reward threshold for an experience to be used as a few-shot example
POSITIVE_REWARD_THRESHOLD = 0.5


@dataclass
class Experience:
    """A single Text-to-SQL experience stored in Qdrant."""

    question: str
    sql: str
    schema_context: str
    reward: float  # 0.0 = wrong, 1.0 = correct
    db_id: str = "default"
    correction: Optional[str] = None  # Corrected SQL if the original was wrong
    metadata: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        """Serialise to Qdrant point payload."""
        return {
            "question": self.question,
            "sql": self.sql,
            "schema_context": self.schema_context,
            "reward": self.reward,
            "db_id": self.db_id,
            "correction": self.correction,
            **self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "Experience":
        known_keys = {"question", "sql", "schema_context", "reward", "db_id", "correction"}
        metadata = {k: v for k, v in payload.items() if k not in known_keys}
        return cls(
            question=payload["question"],
            sql=payload["sql"],
            schema_context=payload.get("schema_context", ""),
            reward=payload.get("reward", 0.0),
            db_id=payload.get("db_id", "default"),
            correction=payload.get("correction"),
            metadata=metadata,
        )


class QdrantMemory:
    """
    Manages Text-to-SQL experiences in a Qdrant vector store.

    Supports both in-memory Qdrant (for testing/local use) and a
    remote Qdrant server (for production).

    The embedding function must return a list of floats with a fixed
    dimensionality that matches ``vector_size``.
    """

    def __init__(
        self,
        embed_fn,
        vector_size: int = 384,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = EXPERIENCES_COLLECTION,
    ) -> None:
        """
        Args:
            embed_fn: Callable ``(text: str) -> List[float]`` that produces
                embeddings for a given piece of text.
            vector_size: Dimensionality of the embedding vectors.
            qdrant_url: URL of a remote Qdrant server.  If *None*, an
                in-process in-memory Qdrant instance is used.
            qdrant_api_key: API key for the remote Qdrant server.
            collection_name: Name of the Qdrant collection to use.
        """
        self.embed_fn = embed_fn
        self.vector_size = vector_size
        self.collection_name = collection_name

        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            logger.info("Connected to remote Qdrant at %s", qdrant_url)
        else:
            self.client = QdrantClient(":memory:")
            logger.info("Using in-memory Qdrant instance")

        self._ensure_collection()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '%s'", self.collection_name)

    def _embed_query(self, question: str, schema_context: str = "") -> List[float]:
        """Embed the concatenation of question and schema context."""
        text = f"{question} {schema_context}".strip()
        return self.embed_fn(text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, experience: Experience) -> str:
        """
        Store an experience in Qdrant and return its point ID.

        Args:
            experience: The :class:`Experience` to store.

        Returns:
            The UUID string used as the point ID.
        """
        point_id = str(uuid.uuid4())
        vector = self._embed_query(experience.question, experience.schema_context)

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=experience.to_payload(),
                )
            ],
        )
        logger.debug(
            "Stored experience (reward=%.2f) for question: %s",
            experience.reward,
            experience.question[:60],
        )
        return point_id

    def retrieve(
        self,
        question: str,
        schema_context: str = "",
        top_k: int = 5,
        min_reward: float = POSITIVE_REWARD_THRESHOLD,
        db_id: Optional[str] = None,
    ) -> List[Experience]:
        """
        Retrieve the *top_k* most-similar successful experiences.

        Args:
            question: The natural-language question to find examples for.
            schema_context: Schema description used together with the question
                for the similarity search.
            top_k: Maximum number of examples to return.
            min_reward: Minimum reward score to be considered a *successful*
                experience.
            db_id: If provided, restrict retrieval to experiences for this DB.

        Returns:
            A list of :class:`Experience` objects sorted by similarity
            (most similar first).
        """
        vector = self._embed_query(question, schema_context)

        # Build filter: reward >= min_reward, optionally filter by db_id
        must_conditions = [
            FieldCondition(key="reward", range=Range(gte=min_reward))
        ]
        if db_id is not None:
            must_conditions.append(
                FieldCondition(key="db_id", match=MatchValue(value=db_id))
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
            query_filter=Filter(must=must_conditions),
            with_payload=True,
        )

        experiences = [
            Experience.from_payload(hit.payload)
            for hit in response.points
            if hit.payload is not None
        ]
        logger.debug("Retrieved %d examples for: %s", len(experiences), question[:60])
        return experiences

    def collection_size(self) -> int:
        """Return the total number of points in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0

    def clear(self) -> None:
        """Delete all points in the collection (useful for tests)."""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
        logger.info("Cleared Qdrant collection '%s'", self.collection_name)

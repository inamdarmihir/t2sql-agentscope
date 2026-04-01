"""
Qdrant-backed episodic memory for the Text-to-SQL RL loop.

Provides:
  - Experience   – dataclass representing one stored (question, SQL, reward) tuple
  - QdrantMemory – vector store wrapper with store / retrieve / clear API
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)


# ---------------------------------------------------------------------------
# Experience dataclass
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    """A single training experience stored in Qdrant."""

    question: str
    sql: str
    schema_context: str
    reward: float
    db_id: str = ""
    correction: Optional[str] = None

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_payload(self) -> dict:
        return {
            "question": self.question,
            "sql": self.sql,
            "schema_context": self.schema_context,
            "reward": self.reward,
            "db_id": self.db_id,
            "correction": self.correction,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "Experience":
        return cls(
            question=payload.get("question", ""),
            sql=payload.get("sql", ""),
            schema_context=payload.get("schema_context", ""),
            reward=float(payload.get("reward", 0.0)),
            db_id=payload.get("db_id", ""),
            correction=payload.get("correction"),
        )


# ---------------------------------------------------------------------------
# QdrantMemory
# ---------------------------------------------------------------------------

_COLLECTION = "t2sql_memory"


class QdrantMemory:
    """
    Vector-store wrapper around Qdrant.

    Args:
        embed_fn:    Callable ``(text: str) -> List[float]``.  Required.
                     For production use ``sentence-transformers``; for tests
                     pass a deterministic hash-based function.
        vector_size: Dimensionality of the embedding vectors.
        host:        Qdrant host.  Pass ``None`` (default) to use an
                     **in-memory** Qdrant instance (no server required).
        port:        Qdrant port (ignored when *host* is ``None``).
        collection_name: Collection to use inside Qdrant.
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        vector_size: int = 384,
        host: Optional[str] = None,
        port: int = 6333,
        collection_name: str = _COLLECTION,
    ) -> None:
        if embed_fn is None:
            embed_fn = _load_default_embed_fn(vector_size)
        self._embed_fn = embed_fn
        self._vector_size = vector_size
        self.collection_name = collection_name

        if host is None:
            # In-memory Qdrant – no external server required
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(
                host=host,
                port=port,
            )

        self._ensure_collection()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size, distance=Distance.COSINE
                ),
            )

    def _embed(self, text: str) -> List[float]:
        return self._embed_fn(text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, experience: Experience) -> str:
        """
        Persist *experience* in Qdrant.

        Returns:
            The UUID string used as the Qdrant point ID.
        """
        point_id = str(uuid.uuid4())
        vector = self._embed(experience.question)
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
        return point_id

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_reward: float = 0.0,
        db_id: Optional[str] = None,
    ) -> List[Experience]:
        """
        Retrieve top-k experiences semantically similar to *query*.

        Args:
            query:      Natural-language question to search for.
            top_k:      Maximum number of results.
            min_reward: Exclude experiences with reward below this threshold.
            db_id:      If provided, restrict results to this database ID.
        """
        vector = self._embed(query)

        must_clauses = []
        if min_reward > 0.0:
            must_clauses.append(
                FieldCondition(key="reward", range=Range(gte=min_reward))
            )
        if db_id:
            must_clauses.append(
                FieldCondition(key="db_id", match=MatchValue(value=db_id))
            )

        query_filter = Filter(must=must_clauses) if must_clauses else None

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=top_k,
            query_filter=query_filter,
        )
        return [Experience.from_payload(hit.payload or {}) for hit in hits]

    def collection_size(self) -> int:
        """Return the number of stored experiences."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0

    def clear(self) -> None:
        """Delete and recreate the collection."""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Legacy helpers (kept for tasks.py T2SQLAgent compatibility)
    # ------------------------------------------------------------------

    def save(self, query: str, sql: str) -> None:
        """Store a bare (query, sql) pair with reward=1.0."""
        self.store(Experience(
            question=query,
            sql=sql,
            schema_context="",
            reward=1.0,
        ))

    def search(self, query: str, limit: int = 2) -> List[Experience]:
        """Search and return Experience objects (legacy thin wrapper)."""
        return self.retrieve(query, top_k=limit)


# ---------------------------------------------------------------------------
# Default embedding function (sentence-transformers, lazy-loaded)
# ---------------------------------------------------------------------------

def _load_default_embed_fn(vector_size: int) -> Callable[[str], List[float]]:
    """
    Return an embedding function backed by sentence-transformers.

    Raises ImportError with a helpful message if the package is missing.
    The model is loaded once and cached.
    """
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required when no embed_fn is provided. "
            "Install it with:  pip install sentence-transformers\n"
            "Or pass embed_fn= to QdrantMemory() to use your own embedder."
        ) from exc

    # all-MiniLM-L6-v2 → 384-d; paraphrase-MiniLM-L3-v2 → 384-d
    model_name = "all-MiniLM-L6-v2" if vector_size == 384 else "paraphrase-MiniLM-L3-v2"
    _model = SentenceTransformer(model_name)

    def _embed(text: str) -> List[float]:
        return _model.encode(text, convert_to_numpy=True).tolist()

    return _embed

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import random

class QdrantMemory:
    def __init__(self, host="localhost", port=6333, collection_name="t2sql_memory"):
        self.collection_name = collection_name
        try:
            self.client = QdrantClient(host=host, port=port)
            self._init_collection()
            self.is_connected = True
        except Exception as e:
            print(f"Warning: Could not connect to Qdrant - {e}. Memory will be disabled.")
            self.is_connected = False

    def _init_collection(self):
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            # Using vector size 384 (e.g. for simple sentence transformers)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def _get_embedding(self, text: str):
        # Placeholder for actual embedding logic. 
        # In a real scenario, use an embedding model like sentence-transformers or AgentScope embedding models.
        # Returning a random 384-d vector for boilerplate purposes.
        return [random.uniform(-1.0, 1.0) for _ in range(384)]

    def save(self, query: str, sql: str):
        if not self.is_connected:
            return
            
        vector = self._get_embedding(query)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"query": query, "sql": sql}
                )
            ]
        )

    def search(self, query: str, limit=2):
        if not self.is_connected:
            return []
            
        vector = self._get_embedding(query)
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=limit
            )
            return [hit.payload for hit in results]
        except Exception:
            return []

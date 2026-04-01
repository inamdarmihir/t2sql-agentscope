"""
Main entry point for the Text-to-SQL agent with Qdrant RL self-improvement.

Usage:
    python main.py --question "What is the average salary per department?"

Environment variables:
    OPENAI_API_KEY   – Required when using OpenAI model configs.
    QDRANT_URL       – Optional remote Qdrant URL (defaults to in-memory).
    QDRANT_API_KEY   – Optional Qdrant API key.
    MODEL_CONFIG     – AgentScope model config name (default: gpt-4o).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import agentscope

from src.agents.feedback_agent import FeedbackAgent
from src.agents.t2sql_agent import T2SQLAgent
from src.database.db_manager import DatabaseManager
from src.memory.qdrant_memory import QdrantMemory
from src.rl_loop import RLLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_embed_fn(model_name: str = "all-MiniLM-L6-v2"):
    """Return a sentence-transformers embedding function."""
    from sentence_transformers import SentenceTransformer  # lazy import

    _model = SentenceTransformer(model_name)

    def embed_fn(text: str):
        return _model.encode(text, convert_to_numpy=True).tolist()

    return embed_fn, _model.get_sentence_embedding_dimension()


def load_model_configs(config_path: str = "config/model_configs.json") -> list:
    with open(config_path) as fh:
        raw = json.load(fh)
    # Expand environment variables
    text = json.dumps(raw)
    for key, val in os.environ.items():
        text = text.replace(f"${{{key}}}", val)
    return json.loads(text)


def build_system(
    model_config_name: str = "gpt-4o",
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    use_llm_feedback: bool = True,
):
    """Initialise and wire all components."""
    # 1. AgentScope
    model_configs = load_model_configs()
    agentscope.init(model_configs=model_configs)
    logger.info("AgentScope initialised with %d model config(s)", len(model_configs))

    # 2. Embeddings + Qdrant memory
    embed_fn, vector_size = build_embed_fn()
    memory = QdrantMemory(
        embed_fn=embed_fn,
        vector_size=vector_size,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )

    # 3. Database (sample data)
    db = DatabaseManager(db_id="sample")
    db.load_sample_data()

    # 4. Agents
    t2sql = T2SQLAgent(
        name="T2SQLAgent",
        model_config_name=model_config_name,
        memory=memory,
        top_k_examples=3,
        db_id="sample",
    )
    feedback = FeedbackAgent(
        name="FeedbackAgent",
        model_config_name=model_config_name,
        use_llm=use_llm_feedback,
    )

    # 5. RL loop
    loop = RLLoop(
        t2sql_agent=t2sql,
        feedback_agent=feedback,
        db_manager=db,
        qdrant_memory=memory,
        store_threshold=0.5,
        apply_corrections=True,
    )
    return loop


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text-to-SQL with AgentScope + Qdrant RL")
    p.add_argument("--question", "-q", type=str, help="Natural-language question to answer")
    p.add_argument(
        "--batch-file",
        type=str,
        help="Path to a plain-text file with one question per line",
    )
    p.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_CONFIG", "gpt-4o"),
        help="AgentScope model config name (default: gpt-4o)",
    )
    p.add_argument("--no-llm-feedback", action="store_true", help="Use execution-only feedback")
    p.add_argument("--qdrant-url", type=str, default=os.getenv("QDRANT_URL"))
    p.add_argument("--qdrant-api-key", type=str, default=os.getenv("QDRANT_API_KEY"))
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.question and not args.batch_file:
        print("Please provide --question or --batch-file.", file=sys.stderr)
        return 1

    loop = build_system(
        model_config_name=args.model,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        use_llm_feedback=not args.no_llm_feedback,
    )

    if args.batch_file:
        with open(args.batch_file) as fh:
            questions = [l.strip() for l in fh if l.strip()]
        results = loop.run_batch(questions)
    else:
        results = [loop.run_once(args.question)]

    for result in results:
        print("\n" + "=" * 70)
        print(result.summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())

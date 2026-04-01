import os

from celery import Celery
from celery.signals import worker_process_init

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "t2sql_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["src.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


@worker_process_init.connect
def _init_agentscope(**kwargs):
    """Initialize AgentScope once per worker process, not per task."""
    import agentscope

    config_path = os.environ.get("AGENTSCOPE_CONFIG", "config/model_configs.json")
    agentscope.init(model_configs=config_path)

# Text-to-SQL with AgentScope, Ollama, Celery, and Qdrant

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![AgentScope](https://img.shields.io/badge/AgentScope-0.1.x-FF6B35?logo=github&logoColor=white)](https://github.com/modelscope/agentscope)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-000000?logo=ollama&logoColor=white)](https://ollama.com)
[![Qdrant](https://img.shields.io/badge/Powered%20by-Qdrant-DC244C?logo=qdrant&logoColor=white)](https://qdrant.tech)
[![Celery](https://img.shields.io/badge/Celery-5.x-37814A?logo=celery&logoColor=white)](https://docs.celeryq.dev)
[![Redis](https://img.shields.io/badge/Redis-7.x-DC382D?logo=redis&logoColor=white)](https://redis.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready boilerplate for multi-agent Text-to-SQL using a fully local stack. Separate agents handle schema understanding, query generation, validation, and execution. No OpenAI. No API costs. Just your machine.

The interesting part: it runs on any quantized model via Ollama, uses Qdrant to store and retrieve past correct queries as few-shot context, and offloads long-running generation to a Celery task queue so nothing blocks.

> This project is built to understand how multi-agent orchestration works with local quantized models. The stack is a starting point — swap in what makes sense for your setup.

---

## Architecture

```
Natural language query
         |
         v
  [Celery Task Queue]  <-- Redis broker
         |
         v
  [Schema Agent]       analyzes table structure and relationships
         |
         v
  [SQL Generator]      queries Qdrant for similar past examples,
                       generates SQL using local Ollama model
         |
         v
  [Validator Agent]    checks syntax and schema compatibility
         |
         v
  [Executor Agent]     runs SQL against SQLite, returns results
         |
         v
  [Feedback + Memory]  stores correct queries back to Qdrant
                       for future few-shot retrieval
```

---

## Stack

| Layer | Package | Version |
|-------|---------|---------|
| 🤖 Agent framework | [AgentScope](https://github.com/modelscope/agentscope) | `^0.1.0` |
| 🧠 Local LLM | [Ollama](https://ollama.com) | Latest |
| 🗄️ Vector memory | [Qdrant](https://qdrant.tech) (`qdrant-client`) | `^1.9.0` |
| ⚡ Task queue | [Celery](https://docs.celeryq.dev) | `^5.3.0` |
| 📨 Message broker | [Redis](https://redis.io) | `^7.0` |
| 🗃️ Database | SQLite | Built-in |
| 🐍 Language | Python | `3.9+` |

---

## Plug in any quantized model

The pipeline is not tied to any specific model. Anything in the Ollama ecosystem works, including `qwen2.5`, `llama3`, `mistral`, `phi3`, `codellama`, or any `.gguf` you pull locally.

**1. Pull your model:**

```bash
ollama run llama3:8b-instruct-q4_K_M
```

**2. Update `config/model_configs.json`:**

```json
{
    "model_type": "ollama_chat",
    "config_name": "ollama_chat_config",
    "model_name": "llama3:8b-instruct-q4_K_M",
    "client_args": {
        "host": "http://127.0.0.1:11434"
    }
}
```

That is all. No agent code changes required.

---

## Prerequisites

- [Docker and Docker Compose](https://docs.docker.com/get-docker/)
- Python 3.9 or later
- [Ollama](https://ollama.com/) running locally

Pull the default model:

```bash
ollama run qwen2.5
```

---

## Setup

### 1. Start infrastructure

Starts Redis (Celery broker) and Qdrant (vector memory):

```bash
docker-compose up -d
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Celery worker

Open a dedicated terminal and keep it running:

```bash
# Linux / macOS
celery -A src.celery_app worker -l info

# Windows
celery -A src.celery_app worker -l info -P eventlet
```

### 4. Run the application

In a separate terminal, dispatch a natural language query:

```bash
python main.py
```

The pipeline runs in sequence: Schema Agent reads the request, SQL Generator pulls similar past queries from Qdrant and generates SQL, Validator checks it, Executor runs it against the local SQLite database, and the result comes back through Celery.

### 5. Run the demo without any external services

```bash
python demo.py
```

The demo monkey-patches AgentScope and uses a hash-based pseudo-embedding so the full RL loop (generate, evaluate, store, retrieve) runs without Ollama, Qdrant, or Redis.

---

## Project structure

```
t2sql-agentscope/
├── main.py                      # Entry point: dispatches task to Celery
├── demo.py                      # Self-contained demo, no external services needed
├── docker-compose.yml           # Redis + Qdrant
├── requirements.txt
├── config/
│   └── model_configs.json       # Ollama model configuration
├── src/
│   ├── agents/
│   │   ├── t2sql_agent.py       # SQL generation with Qdrant few-shot context
│   │   └── feedback_agent.py    # Result evaluation and quality scoring
│   ├── database/
│   │   └── db_manager.py        # SQLite setup and query execution
│   ├── memory/
│   │   └── qdrant_memory.py     # Vector store: upsert, search, collection size
│   ├── rl_loop.py               # Generate, execute, evaluate, store loop
│   ├── tasks.py                 # Celery task definitions
│   └── celery_app.py            # Celery and Redis configuration
└── tests/
```

---

## How the memory loop works

Every query that scores above the `store_threshold` (default `0.5`) gets written to Qdrant with its question, SQL, execution result, and score. The next time a semantically similar question arrives, the T2SQL agent retrieves the top-K past examples and injects them into the generation prompt as few-shot context before calling the model.

The system gets better with use. Not because the model changes, but because the retrieval context improves.

```python
loop = RLLoop(
    t2sql_agent=t2sql,
    feedback_agent=feedback,
    db_manager=db,
    qdrant_memory=memory,
    store_threshold=0.5,   # only store queries scoring above this
    apply_corrections=True,
)
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis broker URL |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `DB_ID` | `sample` | SQLite database identifier |

---

<div align="center">
  <sub>Built with <a href="https://github.com/modelscope/agentscope">AgentScope</a> · <a href="https://ollama.com">Ollama</a> · <a href="https://qdrant.tech">Qdrant</a> · <a href="https://docs.celeryq.dev">Celery</a></sub>
</div>

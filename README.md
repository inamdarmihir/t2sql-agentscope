# 🚀 Text-to-SQL with Quantized Local Models (AgentScope)

So in this case study, the idea is to wire up a proper multi-agent pipeline using AgentScope — where separate agents handle **schema understanding, query generation, validation, and execution** — instead of dumping everything into one prompt.

The interesting part: it runs entirely on a **quantized Qwen model via Ollama**. No OpenAI. No API bills. Just your machine.

---

### 🏗️ Here's what the stack looks like:
- **AgentScope** for orchestrating the agent conversation flow
- **Qwen (quantized)** via Ollama for local inference that's actually fast
- **Qdrant** for vector memory so agents reuse good past queries
- **Celery + Redis** for async task handling when generation runs long

The quantized Qwen angle is what makes this worth exploring right now. The model punches well above its weight at this size, and running it locally with AgentScope's multi-agent routing is a combo I hadn't seen documented anywhere.

*(Source code is 100% open source!)*

👉 **Disclaimer:**
This is built to understand how multi-agent orchestration works with local quantized models. The stack is a starting point — swap what makes sense for your setup.

---

## 🛠️ Prerequisites

1. **Docker & Docker Compose** (for running Redis and Qdrant)
2. **Python 3.9+**
3. **[Ollama](https://ollama.com/)** installed and running locally.
   - Pull the model: `ollama run qwen2.5` (or any `qwen` variant you prefer)

---

## 🚀 Setup & Execution

### 1. Spin up the Infrastructure
Start the Redis broker and Qdrant vector database:
```bash
docker-compose up -d
```

### 2. Install Dependencies
Set up your Python environment:
```bash
pip install -r requirements.txt
```

### 3. Start the Background Worker
Open a terminal and start the Celery worker to listen for and process LLM tasks:
```bash
# On Windows (uses eventlet for concurrency):
celery -A src.celery_app worker -l info -P eventlet

# On Linux/macOS:
celery -A src.celery_app worker -l info
```

### 4. Run the Application
In a new terminal, dispatch a natural language query to the background task queue:
```bash
python main.py
```
*You will see the pipeline in action: The Schema Agent analyzes the request, the SQL Generator creates the query using Qdrant memory, the Validator checks it, and the Executor runs it against the local SQLite DB.*
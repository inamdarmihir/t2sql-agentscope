# 🚀 Production-Ready Text-to-SQL Boilerplate
**Powered by AgentScope, Ollama, Qdrant, and Celery**

Welcome to a robust, scalable, and fully local Text-to-SQL boilerplate! This project demonstrates how to build a multi-agent system that converts natural language to SQL, executes the queries, and learns from past interactions—all while maintaining a production-grade architecture.

---

## 🌟 Why This Boilerplate?

Building AI agent workflows is easy in a notebook, but deploying them to production is hard. This boilerplate bridges that gap by integrating enterprise-grade tools to handle long-running tasks, persistent memory, and local LLM execution safely and efficiently.

### 🏭 Production-Ready Architecture
- **Asynchronous Task Queue:** LLM generation and database execution can be slow. By using **Celery** and **Redis**, agent interactions are decoupled and moved to background workers. This prevents your main application (like a FastAPI web server) from blocking while waiting for the LLM to respond.
- **Fully Local & Private:** Powered by **Ollama** (`llama3`), your database schema and user queries never leave your local machine or VPC. This guarantees complete privacy, zero API costs, and no rate limits.
- **Containerized Infrastructure:** Essential stateful services like Redis (message broker) and Qdrant (vector database) are orchestrated via **Docker Compose**, making environment parity and teardowns a breeze.

### 🧠 The Power of Qdrant (Vector Memory)
A static AI agent makes the same mistakes repeatedly. By integrating **Qdrant**, this boilerplate gives your Text-to-SQL workflow **Few-Shot Episodic Memory**:
- **Contextual Awareness:** Past successful queries are mapped and stored as dense vector embeddings.
- **Dynamic Self-Improvement:** When a user asks a new question, Qdrant performs a semantic similarity search to retrieve the most relevant past queries. The agent seamlessly injects these into its prompt as few-shot examples, drastically improving SQL accuracy and adapting to your specific domain over time.

---

## 🏗️ System Components

1. **AgentScope Orchestration:** The core multi-agent framework managing dialogue flow between specialized sub-agents.
2. **SQLGenerator Agent:** Receives the user query, fetches relevant context from Qdrant, and generates accurate SQL using `llama3`.
3. **DBExecutor Agent (Feedback Loop):** Executes the generated SQL locally, returning execution data or error traces so the system can validate the AI's logic.
4. **Celery Worker:** Wraps the entire agent sequence into a non-blocking background job.

---

## 🛠️ Prerequisites

1. **Docker & Docker Compose** (for running Redis and Qdrant)
2. **Python 3.9+**
3. **[Ollama](https://ollama.com/)** installed and running locally.
   - Pull and start the default model: `ollama run llama3`

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
*You will see the task dispatched to Celery. The main thread will poll for the result while the agents collaborate in the background to generate and execute the SQL, finally returning the complete payload.*

---

## 💡 Next Steps / Customization
- **Expose an API:** Wrap `main.py` in a **FastAPI** or **Flask** endpoint. The architecture is already set up to handle asynchronous web requests!
- **Swap the Database:** Update `src/database/db_manager.py` to securely connect to your production PostgreSQL or MySQL database instead of the provided dummy SQLite file.
- **Upgrade the LLM:** Tweak `config/model_configs.json` to point to a larger local model (`llama3:70b`, `mistral`) or easily switch to remote APIs (like OpenAI) via AgentScope's standard config.
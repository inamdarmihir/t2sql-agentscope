# Text-to-SQL with AgentScope, Ollama, Celery, and Qdrant

This is a boilerplate implementation of a Text-to-SQL workflow leveraging multiple AI agents.

## Architecture
- **AgentScope**: Multi-agent framework managing the conversation flow.
- **Ollama**: Local LLM provider (using `llama3` by default) for code generation.
- **Celery & Redis**: Background task queue processing to handle long-running LLM generation and database execution asynchronously.
- **Qdrant**: Vector database running in Docker, providing conversational and SQL memory (few-shot context) to the agent.
- **SQLite**: Local dummy database initialized with sample data for the Feedback Agent to execute queries against.

## Prerequisites
1. Docker and Docker Compose
2. Python 3.9+
3. [Ollama](https://ollama.com/) running locally with the `llama3` model.
   - Run: `ollama run llama3`

## Setup & Execution

### 1. Start Infrastructure Services
This will start Redis (for Celery) and Qdrant (for Vector Memory):
```bash
docker-compose up -d
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Celery Worker
Open a terminal and run the background worker:
```bash
# On Windows (uses eventlet):
celery -A src.celery_app worker -l info -P eventlet

# On Linux/macOS:
celery -A src.celery_app worker -l info
```

### 4. Run the Main Application
Open another terminal and dispatch a task:
```bash
python main.py
```

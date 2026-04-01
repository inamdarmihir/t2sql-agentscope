# t2sql-agentscope

**Text-to-SQL with [AgentScope](https://github.com/modelscope/agentscope) + [Qdrant](https://qdrant.tech/) RL Self-Improvement**

Convert natural-language questions to SQL queries using a multi-agent pipeline that **improves itself at runtime** via an episodic-memory reinforcement-learning loop backed by Qdrant.

---

## Architecture

```
User question
     │
     ▼
┌─────────────────────────────────────────────────┐
│  RL Self-Improvement Loop (src/rl_loop.py)       │
│                                                   │
│  1. Retrieve top-k similar past successes         │
│     from Qdrant (few-shot demonstrations)         │
│                                                   │
│  2. T2SQLAgent generates SQL using LLM +          │
│     few-shot context                              │
│                                                   │
│  3. SQL is executed against the database          │
│                                                   │
│  4. FeedbackAgent scores the result               │
│     (execution + optional LLM evaluation)         │
│                                                   │
│  5. (question, SQL, reward) stored in Qdrant      │
│     → next query benefits from this experience    │
└─────────────────────────────────────────────────┘
```

### Components

| Module | Description |
|--------|-------------|
| `src/agents/t2sql_agent.py` | AgentScope agent that generates SQL with few-shot retrieval |
| `src/agents/feedback_agent.py` | Evaluates SQL quality; returns reward in [0, 1] |
| `src/memory/qdrant_memory.py` | Stores/retrieves `(question, SQL, reward)` experiences in Qdrant |
| `src/database/db_manager.py` | Schema introspection and SQL execution (SQLite default) |
| `src/rl_loop.py` | Orchestrates the generate→execute→evaluate→store loop |
| `src/utils/helpers.py` | SQL extraction, formatting, and few-shot prompt building |
| `main.py` | CLI entry point |
| `demo.py` | Self-contained runnable demo (no LLM credentials needed) |

### RL Self-Improvement Strategy

The agent uses **episodic memory + retrieval-augmented generation** as its RL mechanism:

- **Reward signal**: `1.0` if SQL executes and returns rows, `0.0` on error, `0.5` for empty results. Optionally refined by LLM semantic evaluation.
- **Policy improvement**: As Qdrant accumulates more successful `(question → SQL)` examples, the `T2SQLAgent` retrieves the most similar ones as few-shot demonstrations, making future queries more accurate.
- **Error correction**: When the `FeedbackAgent` provides a corrected SQL, it is executed and stored with `reward=1.0`, turning each failure into a learning opportunity.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY=sk-...
```

### 3. Run the demo (no API key needed)

```bash
python demo.py
```

### 4. Run a single query

```bash
python main.py --question "What is the average salary per department?"
```

### 5. Run a batch of questions

```bash
python main.py --batch-file questions.txt
```

---

## Configuration

### Model

Pass `--model` to select an AgentScope model config (defined in `config/model_configs.json`):

```bash
python main.py -q "..." --model gpt-3.5-turbo
```

### Remote Qdrant

```bash
export QDRANT_URL=https://your-cluster.qdrant.io
export QDRANT_API_KEY=your-api-key
python main.py -q "..."
```

Or via CLI flags:

```bash
python main.py -q "..." --qdrant-url https://... --qdrant-api-key ...
```

### Execution-only feedback (no LLM evaluation)

```bash
python main.py -q "..." --no-llm-feedback
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests run without an LLM or external Qdrant server (using in-memory Qdrant and stub agents).

---

## Project Structure

```
t2sql-agentscope/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── model_configs.json       # AgentScope LLM configs
├── src/
│   ├── agents/
│   │   ├── t2sql_agent.py       # Text-to-SQL AgentScope agent
│   │   └── feedback_agent.py    # SQL evaluation + reward agent
│   ├── memory/
│   │   └── qdrant_memory.py     # Qdrant experience store
│   ├── database/
│   │   └── db_manager.py        # Schema introspection & SQL execution
│   ├── utils/
│   │   └── helpers.py           # SQL extraction/formatting utilities
│   └── rl_loop.py               # RL self-improvement orchestrator
├── main.py                      # CLI entry point
├── demo.py                      # Runnable demo
└── tests/
    └── test_t2sql.py            # Unit + integration tests
```

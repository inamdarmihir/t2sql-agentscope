import os

from src.celery_app import celery_app
from src.database.db_manager import _is_safe_sql

# AgentScope is initialized once per worker process via the signal in celery_app.py.
# Do NOT call agentscope.init() here.


@celery_app.task
def process_t2sql_request(user_query: str):
    import agentscope
    from agentscope.agents import DialogAgent
    from agentscope.message import Msg

    from src.agents.feedback_agent import FeedbackAgent
    from src.agents.t2sql_agent import T2SQLAgent
    from src.database.db_manager import DatabaseManager

    dialect = os.environ.get("DB_DIALECT", "sqlite")
    conn_str = os.environ.get("DB_CONNECTION_STRING", "sqlite:///app.db")
    db = DatabaseManager(db_id="app", connection_string=conn_str, dialect=dialect)

    schema_context = db.schema_prompt_context()

    # 1. Schema Understanding Agent
    schema_agent = DialogAgent(
        name="SchemaAnalyzer",
        sys_prompt=(
            "You are a database architecture expert. "
            "Analyze the user request and identify the tables and columns needed. "
            f"Schema:\n{schema_context}"
        ),
        model_config_name="ollama_chat_config",
    )

    # 2. Query Generation Agent
    t2sql_agent = T2SQLAgent(
        name="SQLGenerator",
        sys_prompt=(
            f"You are an expert SQL developer. Convert natural language to {dialect.upper()} SQL. "
            "Output ONLY the SQL query — no prose, no markdown fences."
        ),
        model_config_name="ollama_chat_config",
    )

    # 3. Validation Agent
    validator_agent = DialogAgent(
        name="SQLValidator",
        sys_prompt=(
            "You are a strict SQL validator. Review the SQL. "
            "If it contains no destructive operations (DROP, DELETE, UPDATE, INSERT, TRUNCATE, ALTER), "
            "reply with exactly the word VALID. Otherwise explain the problem."
        ),
        model_config_name="ollama_chat_config",
    )

    # 4. Execution / Feedback Agent
    feedback_agent = FeedbackAgent(
        name="DBExecutor",
        model_config_name="ollama_chat_config",
        use_llm=False,
    )

    # Step 1: Schema analysis
    initial_msg = Msg(name="User", content=user_query, role="user")
    schema_analysis = schema_agent(initial_msg)

    # Step 2: SQL generation
    sql_input = Msg(
        name="System",
        content=f"User Query: {user_query}\nSchema Context: {schema_analysis.content}",
        role="user",
    )
    sql_msg = t2sql_agent(sql_input)
    generated_sql = sql_msg.content or ""

    # Step 3: AST-level safety gate (primary) — block before any LLM validation
    if not _is_safe_sql(generated_sql):
        return {
            "query": user_query,
            "schema_analysis": schema_analysis.content,
            "generated_sql": generated_sql,
            "validation_result": "BLOCKED by AST safety check.",
            "execution_result": "Blocked: query contains a disallowed write/DDL keyword.",
        }

    # Step 4: LLM validation (secondary gate)
    validation_msg = validator_agent(sql_msg)
    validation_text = (validation_msg.content or "").strip()
    if "VALID" not in validation_text.upper():
        return {
            "query": user_query,
            "schema_analysis": schema_analysis.content,
            "generated_sql": generated_sql,
            "validation_result": validation_text,
            "execution_result": f"Blocked by validator: {validation_text}",
        }

    # Step 5: Execute only after both gates pass
    result_msg = feedback_agent(sql_msg)

    return {
        "query": user_query,
        "schema_analysis": schema_analysis.content,
        "generated_sql": generated_sql,
        "validation_result": validation_text,
        "execution_result": result_msg.content,
    }

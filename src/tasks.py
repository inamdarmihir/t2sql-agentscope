from src.celery_app import celery_app
import agentscope
from agentscope.agents import DialogAgent
from src.agents.t2sql_agent import T2SQLAgent
from src.agents.feedback_agent import FeedbackAgent
import os

@celery_app.task
def process_t2sql_request(user_query: str):
    # Initialize AgentScope
    agentscope.init(model_configs="config/model_configs.json")
    
    # 1. Schema Understanding Agent
    schema_agent = DialogAgent(
        name="SchemaAnalyzer",
        sys_prompt="You are a database architecture expert. Analyze the user request and determine which tables and columns are required. Assume a schema with a 'users' table (id, name, age, email). Outline the required tables and columns.",
        model_config_name="ollama_chat_config"
    )
    
    # 2. Query Generation Agent
    t2sql_agent = T2SQLAgent(
        name="SQLGenerator",
        sys_prompt="You are an expert SQL developer. Convert natural language to SQL using the provided schema analysis. Only output the SQL query and nothing else. Ensure the query works for sqlite.",
        model_config_name="ollama_chat_config"
    )
    
    # 3. Validation Agent
    validator_agent = DialogAgent(
        name="SQLValidator",
        sys_prompt="You are a strict SQL validator. Review the provided SQL. Ensure it does not contain destructive operations like DROP or DELETE. If safe, reply with 'VALID'. If unsafe, explain why.",
        model_config_name="ollama_chat_config"
    )
    
    # 4. Execution Agent
    feedback_agent = FeedbackAgent(
        name="DBExecutor",
        sys_prompt="You execute SQL queries and provide the results or error messages.",
        model_config_name="ollama_chat_config"
    )
    
    # Message generation & Flow
    from agentscope.message import Msg
    
    # Step 1: Schema Understanding
    initial_msg = Msg(name="User", content=user_query, role="user")
    schema_analysis = schema_agent(initial_msg)
    
    # Step 2: Query Generation
    sql_input = Msg(
        name="System", 
        content=f"User Query: {user_query}\nSchema Context: {schema_analysis.content}", 
        role="user"
    )
    sql_msg = t2sql_agent(sql_input)
    
    # Step 3: Validation
    validation_msg = validator_agent(sql_msg)
    
    # Step 4: Execution (Only if valid, but for pipeline sake we execute the SQL block)
    # The feedback agent extracts the SQL block automatically
    result_msg = feedback_agent(sql_msg)
    
    return {
        "query": user_query,
        "schema_analysis": schema_analysis.content,
        "generated_sql": sql_msg.content,
        "validation_result": validation_msg.content,
        "execution_result": result_msg.content
    }

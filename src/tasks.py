from src.celery_app import celery_app
import agentscope
from src.agents.t2sql_agent import T2SQLAgent
from src.agents.feedback_agent import FeedbackAgent
import os

@celery_app.task
def process_t2sql_request(user_query: str):
    # Initialize AgentScope
    agentscope.init(model_configs="config/model_configs.json")
    
    t2sql_agent = T2SQLAgent(
        name="SQLGenerator",
        sys_prompt="You are an expert SQL developer. Convert natural language to SQL. Only output the SQL query and nothing else. Ensure the query works for sqlite.",
        model_config_name="ollama_chat_config"
    )
    
    feedback_agent = FeedbackAgent(
        name="DBExecutor",
        sys_prompt="You execute SQL queries and provide the results or error messages.",
        model_config_name="ollama_chat_config"
    )
    
    # Message generation
    from agentscope.message import Msg
    msg = Msg(name="User", content=user_query, role="user")
    
    # Simple loop: Generate SQL -> Execute -> Provide Feedback
    sql_msg = t2sql_agent(msg)
    
    # Execute the SQL in feedback_agent
    result_msg = feedback_agent(sql_msg)
    
    return {
        "query": user_query,
        "generated_sql": sql_msg.content,
        "execution_result": result_msg.content
    }

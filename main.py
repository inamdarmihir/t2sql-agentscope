from src.tasks import process_t2sql_request
import time

def main():
    print("Welcome to Text-to-SQL Agent Workflow")
    print("Ensure Redis and Qdrant are running via docker-compose.")
    print("Ensure Ollama is running 'llama3' locally.")
    print("Ensure Celery worker is running.")
    print("-" * 50)
    
    query = "Find all users older than 25 and show their email addresses"
    print(f"Submitting Query: '{query}' to Celery...")
    
    # Submitting task to celery
    task = process_t2sql_request.delay(query)
    
    print(f"Task ID: {task.id}")
    print("Waiting for result...")
    
    # Wait for result synchronously (for demo purposes)
    while not task.ready():
        time.sleep(1)
        
    print("-" * 50)
    print("Task completed!")
    result = task.result
    
    print("\n[Generated SQL]:")
    print(result.get("generated_sql", ""))
    
    print("\n[Execution Result]:")
    print(result.get("execution_result", ""))

if __name__ == "__main__":
    main()

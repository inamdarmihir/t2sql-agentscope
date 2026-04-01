from agentscope.agents import DialogAgent
from src.memory.qdrant_memory import QdrantMemory
from agentscope.message import Msg

class T2SQLAgent(DialogAgent):
    def __init__(self, name, sys_prompt, model_config_name):
        super().__init__(name=name, sys_prompt=sys_prompt, model_config_name=model_config_name)
        self.memory_db = QdrantMemory()
        
    def reply(self, x=None):
        original_content = x.content if x and x.content else ""
        
        # Fetch similar past queries
        if original_content:
            context = self.memory_db.search(original_content)
            if context:
                x.content = f"Context from past queries:\n{context}\n\nCurrent Request: {original_content}"
        
        # Get response from the LLM
        response = super().reply(x)
        
        # Restore original content for next agents if needed
        if original_content:
            x.content = original_content
            
        # Save to memory (in background or directly)
        if original_content and response.content:
            self.memory_db.save(original_content, response.content)
            
        return response

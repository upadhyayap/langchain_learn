from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, List, Any
from langchain.schema import LLMResult

class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs
        ) -> Any:
        print(f"Prompt to LLM was: {prompts[0]}")
        print("================")
        print(f"Running LLM... with input: {kwargs.get('input')}")
        print("================")

    def on_llm_end(
            self, response: LLMResult, **kwargs
        ) -> Any:
        print(f"LLM response: {response}")
        print("================")
        print(f"LLM End: {response}")
        print("================")

    def on_agent_action(self, action, **kwargs):
        print(f"Agent Action: {action}")
    
    def on_agent_finish(self, finish, **kwargs):
        print(f"Agent Finish: {finish}")
        
    def on_tool_start(self, tool, **kwargs):
        print(f"Tool Start: {tool}")
        
    def on_tool_end(self, output, **kwargs):
        print(f"Tool End: {output}")
        
from langchain.agents import Tool, initialize_agent, AgentExecutor, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from typing import Union, List

# Define your custom output parser for the three-part format
class ThreePartOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        # Check if the agent is still thinking through the problem
        if "Action:" in text:
            action_match = re.search(r"Action: (.*?)[\n]", text)
            action_input_match = re.search(r"Action Input: (.*)", text)
            
            if not action_match or not action_input_match:
                raise ValueError(f"Could not parse action and action input from text: {text}")
            
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            
            return AgentAction(tool=action, tool_input=action_input, log=text)
        
        # If no more actions, then the agent is done
        return AgentFinish(return_values={"output": text}, log=text)
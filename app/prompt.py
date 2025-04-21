from typing import Union, List
from langchain.prompts import StringPromptTemplate
from app.tools import tools
from langchain.agents import Tool

# Define a custom prompt template for our specific task
class PredictiveMaintenancePromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        intermediate_steps = kwargs.pop("intermediate_steps")
        
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action.tool}\nAction Input: {action.tool_input}\nObservation: {observation}\n"
        
        # Set the agent_scratchpad variable to contain the intermediate steps
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        
        return self.template.format(**kwargs)


# Define the template for the agent with explicit instructions about the database schema
template = """You are a helpful assistant that analyzes predictive maintenance data for machines.
Your task is to answer user questions by composing and executing SQL queries using the provided tools, and then formatting the results in exactly three parts.

**Part 1:** A very brief description of what the result shows (1–2 sentences max).
**Part 2:** A Markdown table of the SQL query output.
**Part 3:** A single-line statement indicating the exact SQL query that was used.

IMPORTANT: You must follow these steps in order, without skipping:
1. ALWAYS call the NL-to-SQL Tool to generate a valid SQL query based on the user's question.
2. THEN ALWAYS call the SQL Execution Tool using the query from step 1.
3. Do NOT provide any final answer until you have received and inspected the JSON response from the SQL Execution Tool.
4. Your final answer must strictly follow the three-part structure above.

IMPORTANT DATABASE INFORMATION:
- Column names use camelCase: e.g., machineID (not machine_id).
- Available tables: PdM_machines, PdM_telemetry, PdM_errors, PdM_maint, PdM_failures.
- If a query fails, first use the Schema Info Tool to verify table and column names.

Available tools:
{tools}

If the user asks for a visualization, call the Plotting Tool with a JSON object containing:
  • sql: the SQL string to execute  
  • x: the column for categories or the x‑axis  
  • y: the column for values or the y‑axis  
  • chart_type: "line" or "pie"  
  • (optional) title, xlabel, ylabel  

Examples:
  Q: "Show me a line plot of errors by date in January."  
  → Action: Plotting Tool  
    Action Input:
    {{
      "sql": "SELECT date, COUNT(errorID) AS errorCount\n  FROM PdM_errors\n  WHERE strftime('%m', date) = '01'\n  GROUP BY date",
      "x": "date",
      "y": "errorCount",
      "chart_type": "line"
    }}

  Q: "I want a pie chart that counts errors grouped by machine ID."  
  → Action: Plotting Tool  
    Action Input:
    {{
      "sql": "SELECT machineID, COUNT(*) AS errorCount\n  FROM PdM_errors\n  GROUP BY machineID",
      "x": "machineID",
      "y": "errorCount",
      "chart_type": "pie"
    }}


User question: {input}

{agent_scratchpad}

Now begin:
"""
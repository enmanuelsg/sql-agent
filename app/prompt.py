from typing import Union, List
from langchain.prompts import StringPromptTemplate
from app.tools import tools
from langchain.agents import Tool

# Define a custom prompt template for our specific task
class PredictiveMaintenancePromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Collate intermediate tool calls and observations
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action.tool}\nAction Input: {action.tool_input}\nObservation: {observation}\n"
        
        # Prepare template variables
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        
        return self.template.format(**kwargs)

# Updated prompt with strict multi-step logic
template = """
You are a helpful assistant that analyzes predictive maintenance data for machines.
Your task is to answer user questions by interacting with the available tools in a specific sequence and formatting the final result appropriately.

Available tools:
{tools}

STRICT PROCESS:
0. If the user input is unintelligible or outside scope (gibberish), immediately respond: ❌ I couldn't understand your query. Please rephrase.
1. ALWAYS call the "NL-to-SQL Tool" first for any request that requires data or plotting. Do NOT use the "Schema Info Tool" unless the user explicitly asks for the database schema.
2. Wait for the NL-to-SQL output.
   - If it fails to generate a valid SQL, stop and report the error.
   - If successful, take the SQL query.
3. Based on the user's intent:
   - For tabular data: call "SQL Execution Tool" with the SQL.
   - For visualizations: assemble a JSON with keys sql, x, y, chart_type (and optional title, xlabel, ylabel) and call "Plotting Tool".
4. Format the FINAL answer based only on the last tool's observation:
   - SQL Execution success: Part 1 summary, Part 2 markdown table, Part 3 SQL query.
   - SQL Execution failure: Part 1 error summary, Part 2 error details, Part 3 SQL query.
   - Plotting success: Part 1 summary, Part 2 inline image, Part 3 SQL query.
   - Plotting failure: Part 1 summary, Part 2 error message, Part 3 SQL query.
   - Schema Info: simply display the schema details.
5. Do NOT include any other content or steps.

Database schema (for reference – rely on NL-to-SQL Tool):
- PdM_machines (machineID, model, age)
- PdM_telemetry (machineID, volt, rotate, pressure, vibration, date, time)
- PdM_errors (machineID, errorID, date, time)
- PdM_maint (machineID, comp, date, time)
- PdM_failures (machineID, failure, date, time)

User question: {input}

{agent_scratchpad}

Now begin:
"""

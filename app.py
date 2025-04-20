# app.py
import pandas as pd
from config import OPENAI_MODEL_NAME, OPENAI_TEMPERATURE
from config import PLOT_OUTPUT_DIR, DEFAULT_PLOT_FILENAME

OUTPUT_PATH = str(PLOT_OUTPUT_DIR / DEFAULT_PLOT_FILENAME)

import chainlit as cl
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import StringPromptTemplate
from langchain.agents import Tool, initialize_agent, AgentExecutor, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from typing import Union, List
import re
import json
import ast

from config import OPENAI_MODEL_NAME, OPENAI_TEMPERATURE
from app.tools import tools
from utils.db_utils import get_schema_info


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


prompt = PredictiveMaintenancePromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)

# Create a parsing function that ensures responses follow the three-part structure
def format_final_response(agent_output):
    if isinstance(agent_output, dict) and "output" in agent_output:
        output = agent_output["output"]
        
        # If the output already contains markdown table and SQL query info
        if "**Part 1:**" in output and "**Part 2:**" in output and "**Part 3:**" in output:
            return output
        
        # If SQL execution tool was used and returned a string representation of a JSON
        try:
            # Check if there's a JSON string in the output
            json_start = output.find("{")
            json_end = output.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = output[json_start:json_end]
                # Try to parse the JSON
                try:
                    result_dict = json.loads(json_str)
                except:
                    # Try using ast.literal_eval as a fallback
                    result_dict = ast.literal_eval(json_str)
                
                if isinstance(result_dict, dict):
                    # If it's a schema request, format appropriately
                    if "schema_info" in result_dict:
                        schema_desc = "Here is the database schema information."
                        schema_details = str(result_dict["schema_info"])
                        return f"""**Part 1:** {schema_desc}
**Part 2:** 
```
{schema_details}
```

**Part 3:** Schema information retrieved."""
                    
                    # Handle regular query results
                    if "success" in result_dict:
                        if result_dict["success"]:
                            if "query" in result_dict and "markdown_table" in result_dict:
                                # Extract the table name from the query for description
                                query = result_dict["query"]
                                table_match = re.search(r"FROM\s+(\w+)", query, re.IGNORECASE)
                                table_name = table_match.group(1) if table_match else "the database"
                                
                                # Create description based on the query type
                                if "COUNT" in query.upper():
                                    description = f"This shows the count from {table_name}."
                                elif "WHERE" in query.upper():
                                    description = f"This shows filtered results from {table_name}."
                                else:
                                    description = f"This shows data from {table_name}."
                                
                                # Format the three-part response
                                return f"""**Part 1:** {description}
**Part 2:** 
{result_dict["markdown_table"]}

**Part 3:** Used query: {result_dict["query"]}"""
                        else:
                            # Handle failed queries with error information
                            error_message = result_dict.get("error", "Unknown error")
                            return f"""**Part 1:** The query could not be executed due to an error.
**Part 2:** 
```
Error: {error_message}
```

**Part 3:** Failed query: {result_dict.get("query", "Unknown query")}"""
        except Exception as e:
            print(f"Error parsing result: {e}")
        
        # Default format if we can't parse properly
        return f"""**Part 1:** Here are the results of your query.
**Part 2:** 
Unable to format results as a table.

**Part 3:** Query information not available."""
    
    return "I couldn't process your request properly. Please try rephrasing your question."

# Initialize LangChain agent
#llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
llm = ChatOpenAI(
    model_name=OPENAI_MODEL_NAME,
    temperature=OPENAI_TEMPERATURE
)
agent = initialize_agent(
    tools, 
    llm, 
    agent="zero-shot-react-description", 
    verbose=True,
    prompt=prompt,
    return_intermediate_steps=True,
)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("agent", agent)
    
    # Get schema info at the start to help with debugging
    schema_info = get_schema_info()
    print("Database Schema Information:")
    print(schema_info)

@cl.on_message
async def on_message(message: cl.Message):
    agent: AgentExecutor = cl.user_session.get("agent")

    # 1) Run the agent and collect all steps
    result = agent({"input": message.content})
    steps = result["intermediate_steps"]  # list of (AgentAction, observation)

    # 2) Handle Plotting Tool responses, preserving three-part structure
    for action, obs in steps:
        if action.tool == "Plotting Tool":
            # Extract SQL from the tool input JSON
            params = json.loads(action.tool_input)
            sql = params.get("sql", "")

            # Build Part 1 description
            m = re.search(r"WHERE\s+machineID\s*=\s*(\d+)", sql, re.IGNORECASE)
            desc = f"These are the errors for machine id {m.group(1)}." if m else "Here are your results."

            # Part 1 – Summary
            await cl.Message(content=f"**Summary:** {desc}").send()

            # Part 2 – Plot image or error
            if isinstance(obs, str) and obs.lower().endswith(".png"):
                plot_el = cl.Image(path=obs, name="plot", display="inline")
                await cl.Message(
                    content="**Plot:**",
                    elements=[plot_el]
                ).send()
            else:
                await cl.Message(content=f"❌ Plotting Tool error:\n{obs}").send()

            # Part 3 – SQL query used
            await cl.Message(content=f"**Used query:** {sql}").send()
            return

    # 3) Handle SQL Execution Tool responses (table output)
    sql_json = None
    for action, obs in steps:
        if action.tool == "SQL Execution Tool":
            sql_json = json.loads(obs)
            break

    if not sql_json:
        await cl.Message(content="❌ I never executed the SQL tool!").send()
        return

    # Build description for Part 1
    m = re.search(r"WHERE\s+machineID\s*=\s*(\d+)", sql_json["query"], re.IGNORECASE)
    desc = f"These are the errors for machine id {m.group(1)}." if m else "Here are your results."

    # Compose the three parts
    part1 = f"**Summary:** {desc}"
    part2 = f"**Table:**\n{sql_json['markdown_table']}"
    part3 = f"**Used query:** {sql_json['query']}"

    # Send all three parts in one message
    await cl.Message(content="\n\n".join([part1, part2, part3])).send()

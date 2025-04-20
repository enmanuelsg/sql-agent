# app.py
import pandas as pd
from config import OPENAI_MODEL_NAME, OPENAI_TEMPERATURE
from config import PLOT_OUTPUT_DIR, DEFAULT_PLOT_FILENAME

OUTPUT_PATH = str(PLOT_OUTPUT_DIR / DEFAULT_PLOT_FILENAME)

import chainlit as cl
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentExecutor, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from typing import Union, List
import re
import json
import ast

from config import OPENAI_MODEL_NAME, OPENAI_TEMPERATURE
from app.tools import tools
from app.prompt import PredictiveMaintenancePromptTemplate, template
from utils.db_utils import get_schema_info


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


async def _process_agent_response(agent: AgentExecutor, user_input: str):
    """Invoke the LangChain agent and return the list of (action, observation)."""
    result = agent({"input": user_input})
    return result["intermediate_steps"]

async def _handle_plotting_response(action: AgentAction, observation: str):
    """Send the three-part plotting response back to the user."""
    params = json.loads(action.tool_input)
    sql = params.get("sql", "")
    # Part 1 – summary
    m = re.search(r"WHERE\s+machineID\s*=\s*(\d+)", sql, re.IGNORECASE)
    desc = f"These are the errors for machine id {m.group(1)}." if m else "Here are your results."
    await cl.Message(content=f"**Summary:** {desc}").send()

    # Part 2 – plot image or error
    if isinstance(observation, str) and observation.lower().endswith(".png"):
        plot_el = cl.Image(path=observation, name="plot", display="inline")
        await cl.Message(content="**Plot:**", elements=[plot_el]).send()
    else:
        await cl.Message(content=f"❌ Plotting Tool error:\n{observation}").send()

    # Part 3 – the SQL query
    await cl.Message(content=f"**Used query:** {sql}").send()

async def _handle_sql_response(action: AgentAction, observation: str):
    """Send the three-part SQL table response back to the user."""
    sql_json = json.loads(observation)
    # Part 1 – summary
    query = sql_json.get("query", "")
    m = re.search(r"WHERE\s+machineID\s*=\s*(\d+)", query, re.IGNORECASE)
    desc = f"These are the errors for machine id {m.group(1)}." if m else "Here are your results."
    part1 = f"**Summary:** {desc}"

    # Part 2 – table
    table = sql_json.get("markdown_table", "No results returned.")
    part2 = f"**Table:**\n{table}"

    # Part 3 – query
    part3 = f"**Used query:** {query}"

    await cl.Message(content="\n\n".join([part1, part2, part3])).send()

@cl.on_message
async def on_message(message: cl.Message):
    agent: AgentExecutor = cl.user_session.get("agent")

    # 1) Invoke agent
    steps = await _process_agent_response(agent, message.content)

    # 2) If any plotting action, handle and return immediately
    for action, obs in steps:
        if action.tool == "Plotting Tool":
            await _handle_plotting_response(action, obs)
            return

    # 3) Else if any SQL action, handle and return
    for action, obs in steps:
        if action.tool == "SQL Execution Tool":
            await _handle_sql_response(action, obs)
            return

    # 4) No recognized tool run
    await cl.Message(content="❌ I never executed the SQL tool!").send()

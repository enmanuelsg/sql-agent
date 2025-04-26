import re
import json

import chainlit as cl
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentExecutor
from langchain.schema import AgentAction

from config import OPENAI_MODEL_NAME, OPENAI_TEMPERATURE
from app.tools import tools
from app.prompt import PredictiveMaintenancePromptTemplate, template
from utils.db_utils import get_schema_info

# Initialize prompt and agent
prompt = PredictiveMaintenancePromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)

llm = ChatOpenAI(
    model_name=OPENAI_MODEL_NAME,
    temperature=OPENAI_TEMPERATURE,
    streaming=True
)
# Enable parsing errors to be surfaced
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    prompt=prompt,
    return_intermediate_steps=True,
    handle_parsing_errors=True
)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("agent", agent)
    schema_info = get_schema_info()
    print("Database Schema Information:")
    print(schema_info)

async def _process_agent_response(agent: AgentExecutor, user_input: str):
    try:
        result = await agent.acall({"input": user_input})
        return result.get("intermediate_steps", [])
    except Exception as e:
        # Send parsing or execution errors back to the user
        await cl.Message(content=f"❌ Agent error: {e}").send()
        return []

async def _handle_plotting_response(action: AgentAction, observation: str):
    params = json.loads(action.tool_input)
    sql = params.get("sql", "")
    m = re.search(r"WHERE\s+machineID\s*=\s*(\d+)", sql, re.IGNORECASE)
    desc = f"Pie chart of errors for machine id {m.group(1)}." if m else "Generated pie chart."
    await cl.Message(content=f"**Part 1:** {desc}").send()

    if isinstance(observation, str) and observation.lower().endswith(".png"):
        img = cl.Image(path=observation, name="plot", display="inline")
        await cl.Message(content="**Part 2:**", elements=[img]).send()
    else:
        await cl.Message(content=f"**Part 2:** Plotting error: {observation}").send()

    await cl.Message(content=f"**Part 3:** Used query: {sql}").send()

async def _handle_sql_response(action: AgentAction, observation: str):
    sql_json = json.loads(observation)
    query = sql_json.get("query", "")
    m = re.search(r"WHERE\s+machineID\s*=\s*(\d+)", query, re.IGNORECASE)
    desc = f"Results for machine id {m.group(1)}." if m else "Query results."
    part1 = f"**Part 1:** {desc}"
    table = sql_json.get("markdown_table", "No results.")
    part2 = f"**Part 2:**\n{table}"
    part3 = f"**Part 3:** Used query: {query}"
    await cl.Message(content="\n\n".join([part1, part2, part3])).send()

@cl.on_message
async def on_message(message: cl.Message):
    agent: AgentExecutor = cl.user_session.get("agent")

    # Ejecuta el agente con streaming y logging de tools en la UI
    result = await agent.acall(
        {"input": message.content},
        callbacks=[cl.LangchainCallbackHandler(stream_final_answer=True)]
    )

    steps = result.get("intermediate_steps", [])
    if not steps:
        return

    # 1) Check plotting
    for action, obs in steps:
        if action.tool == "Plotting Tool":
            await _handle_plotting_response(action, obs)
            return

    # 2) Check SQL
    for action, obs in steps:
        if action.tool == "SQL Execution Tool":
            await _handle_sql_response(action, obs)
            return

    # 3) Fallback
    await cl.Message(
        content="❌ I never executed a recognized tool. Please rephrase your query or ask for schema info."
    ).send()
import os
import re
import json
import logging

import agentops
from agentops.integration.callbacks.langchain import LangchainCallbackHandler

# ─── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("agentops").setLevel(logging.DEBUG)

# ─── Initialize AgentOps BEFORE importing LLMs ─────────────────────────────────
API_KEY = os.getenv("AGENTOPS_API_KEY")
if not API_KEY:
    raise ValueError("AGENTOPS_API_KEY environment variable is not set or empty")

# Initialize AgentOps with explicit configuration
agentops.init(
    api_key=API_KEY,
    instrument_llm_calls=False,  # disable auto patching; rely on callback handler
    tags=["SQL-AGENT"]
)
print("AgentOps initialized successfully")

# ─── Shared AgentOps Callback Handler ──────────────────────────────────────────
handler = LangchainCallbackHandler(
    api_key=API_KEY,
    tags=["SQL-AGENT"]
)

# ─── Now import and configure your agent dependencies ─────────────────────────
import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentExecutor
from langchain.schema import AgentAction

from config import OPENAI_MODEL_NAME, OPENAI_TEMPERATURE
from app.tools import tools
from app.prompt import PredictiveMaintenancePromptTemplate, template
from utils.db_utils import get_schema_info

# ─── Prompt & Agent Initialization ─────────────────────────────────────────────
prompt = PredictiveMaintenancePromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)

llm = ChatOpenAI(
    model_name=OPENAI_MODEL_NAME,
    temperature=OPENAI_TEMPERATURE,
    streaming=True,
    callbacks=[handler]  # instrument all LLM calls via callback
)

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    prompt=prompt,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    callbacks=[handler]  # instrument agent orchestration
)

# ─── Chainlit Event Handlers ──────────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("agent", agent)
    schema = get_schema_info()
    print("Database schema loaded:", schema)

async def _handle_plotting_response(
    action: AgentAction,
    observation: str,
    final_answer: str
):
    params = json.loads(action.tool_input)
    sql = params.get("sql", "")

    await cl.Message(content=f"**Summary:** {final_answer}").send()
    if isinstance(observation, str) and observation.lower().endswith(".png"):
        img = cl.Image(path=observation, name="plot", display="inline")
        await cl.Message(content="**Result:**", elements=[img]).send()
    else:
        await cl.Message(content=f"**Result:** Plotting error: {observation}").send()
    await cl.Message(content=f"**SQL Used:** {sql}").send()

async def _handle_sql_response(
    action: AgentAction,
    observation: str,
    final_answer: str
):
    result = json.loads(observation)
    table = result.get("markdown_table", "No results.")
    query = result.get("query", "")

    await cl.Message(content=f"**Summary:** {final_answer}").send()
    await cl.Message(content=f"**Result:**\n{table}").send()
    await cl.Message(content=f"**SQL Used:** {query}").send()

@cl.on_message
async def on_message(message: cl.Message):
    agent: AgentExecutor = cl.user_session.get("agent")

    # Execute agent with both AgentOps callback and Chainlit UI callback
    result = await agent.acall(
        {"input": message.content},
        callbacks=[
            handler,
            cl.LangchainCallbackHandler(stream_final_answer=False)
        ]
    )

    steps = result.get("intermediate_steps", [])
    final_answer = result.get("output", "").strip()
    if not steps:
        return

    for action, obs in steps:
        if action.tool == "Plotting Tool":
            await _handle_plotting_response(action, obs, final_answer)
            return
        if action.tool == "SQL Execution Tool":
            await _handle_sql_response(action, obs, final_answer)
            return

    await cl.Message(content="❌ No recognized tool executed.").send()

@cl.on_chat_end
async def on_chat_end():
    # End session so spans flush to dashboard
    agentops.end_session("Success")

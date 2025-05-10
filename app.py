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

from langsmith import Client
#from langsmith.wrappers import OpenAIAgentsTracingProcessor
from langchain.callbacks.tracers import LangChainTracer

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

# —– LangSmith tracing: attach the OpenAI Agents tracing processor
client = Client()  # reads LANGCHAIN_API_KEY, ENDPOINT, PROJECT from your .env
tracer = LangChainTracer(client=client)

# Enable parsing errors to be surfaced
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    prompt=prompt,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    callbacks=[tracer]
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

async def _handle_plotting_response(
    action: AgentAction,
    observation: str,
    final_answer: str
):
    params = json.loads(action.tool_input)
    sql = params.get("sql", "")

    # Usamos el final_answer como Summary
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
    sql_json = json.loads(observation)
    query = sql_json.get("query", "")
    table = sql_json.get("markdown_table", "No results.")

    # Usamos el final_answer como Summary
    await cl.Message(content=f"**Summary:** {final_answer}").send()
    await cl.Message(content=f"**Result:**\n{table}").send()
    await cl.Message(content=f"**SQL Used:** {query}").send()

@cl.on_message
async def on_message(message: cl.Message):
    agent: AgentExecutor = cl.user_session.get("agent")

    # Ejecuta el agente con streaming y logging de tools en la UI
    result = await agent.acall(
        {"input": message.content},
        callbacks=[cl.LangchainCallbackHandler(stream_final_answer=False)]
    )

    steps = result.get("intermediate_steps", [])
    final_answer = result.get("output", "").strip()
    if not steps:
        return

    # 1) Check plotting
    for action, obs in steps:
        if action.tool == "Plotting Tool":
            await _handle_plotting_response(action, obs, final_answer)
            return

    # 2) Check SQL
    for action, obs in steps:
        if action.tool == "SQL Execution Tool":
            await _handle_sql_response(action, obs, final_answer)
            return

    # 3) Fallback
    await cl.Message(
        content="❌ I never executed a recognized tool. Please rephrase your query or ask for schema info."
    ).send()
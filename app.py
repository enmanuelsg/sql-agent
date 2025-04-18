import chainlit as cl
from langchain.agents import Tool, initialize_agent
from langchain_community.chat_models import ChatOpenAI
from app.nl_to_sql import convert_nl_to_sql
from utils.db_utils import execute_query

# Define tools
def nl_to_sql_tool(input_text: str) -> str:
    return convert_nl_to_sql(input_text)

def execute_sql_tool(sql_query: str) -> str:
    return execute_query(sql_query)

tools = [
    Tool(
        name="NL-to-SQL Tool",
        func=nl_to_sql_tool,
        description="Converts a natural language query about predictive maintenance into a SQL query."
    ),
    Tool(
        name="SQL Execution Tool",
        func=execute_sql_tool,
        description="Executes a SQL query on the predictive maintenance database and returns the results."
    )
]

# Initialize LangChain agent
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("agent", agent)

@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    response = agent.run(message.content)
    await cl.Message(content=response).send()

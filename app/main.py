# app/main.py

import sys
import os

import openai

# Patch missing attributes in OpenAI module
if not hasattr(openai, 'Timeout'):
    try:
        openai.Timeout = openai.error.Timeout
    except AttributeError:
        openai.Timeout = Exception

if not hasattr(openai, 'APIConnectionError'):
    try:
        openai.APIConnectionError = openai.error.APIConnectionError
    except AttributeError:
        openai.APIConnectionError = Exception

if not hasattr(openai, 'RateLimitError'):
    try:
        openai.RateLimitError = openai.error.RateLimitError
    except AttributeError:
        openai.RateLimitError = Exception

if not hasattr(openai, 'ServiceUnavailableError'):
    try:
        openai.ServiceUnavailableError = openai.error.ServiceUnavailableError
    except AttributeError:
        openai.ServiceUnavailableError = Exception

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from langchain.agents import Tool, initialize_agent
from langchain_community.chat_models import ChatOpenAI  # using updated import per deprecation warnings
from nl_to_sql import convert_nl_to_sql  # your NL-to-SQL function
from utils.db_utils import execute_query     # your DB execution function

# Define a tool to convert natural language query to SQL
def nl_to_sql_tool(input_text: str) -> str:
    sql_query = convert_nl_to_sql(input_text)
    return sql_query

# Define a tool to execute the SQL query against your SQLite database
def execute_sql_tool(sql_query: str) -> str:
    results = execute_query(sql_query)
    return results

# Wrap the functions as LangChain Tools
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

# Initialize the LangChain agent with an OpenAI LLM.
llm = ChatOpenAI(temperature=0, model_name="gpt-4-0613")
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

def main():
    st.title("Predictive Maintenance Chatbot")
    st.write("Ask questions about machine status, maintenance history, alerts, and more!")
    
    user_query = st.text_input("Enter your query:")
    
    if user_query:
        response = agent.run(user_query)
        st.write("Response:")
        st.write(response)

if __name__ == "__main__":
    main()

# app/tools.py

import json
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

from langchain.agents import Tool

from app.nl_to_sql import convert_nl_to_sql
from utils.db_utils import execute_query

def nl_to_sql_tool(input_text: str) -> str:
    """
    Converts a natural language query about predictive maintenance into a SQL query.
    """
    return convert_nl_to_sql(input_text)

def execute_sql_tool(sql_query: str) -> str:
    """
    Executes a SQL query on the predictive maintenance database and returns the results
    as a JSON string including markdown formatting.
    """
    result = execute_query(sql_query)
    return json.dumps(result)

def plot_tool(params: str) -> Figure:
    """
    Given a JSON string with keys:
      - sql: string, the SQL to run
      - x: string, column for x-axis or categories
      - y: string, column for values
      - chart_type: optional, "line" or "pie"
      - title, xlabel, ylabel: optional strings

    Runs the query, builds an interactive Plotly figure in memory, and returns it.
    """
    args = json.loads(params)

    # 1. Execute SQL and load into DataFrame
    result = execute_query(args["sql"])
    records = result.get("data", [])
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No data returned for plotting.")

    # 2. Build Plotly figure based on chart_type
    chart_type = args.get("chart_type", "line")
    if chart_type == "line":
        fig = px.line(
            df,
            x=args.get("x"),
            y=args.get("y"),
            title=args.get("title"),
            labels={
                args.get("x"): args.get("xlabel", args.get("x")),
                args.get("y"): args.get("ylabel", args.get("y")),
            },
        )
        fig.update_traces(mode="markers+lines")
    elif chart_type == "pie":
        fig = px.pie(
            df,
            names=args.get("x"),
            values=args.get("y"),
            title=args.get("title"),
        )
    else:
        raise ValueError(f"Unsupported chart_type: {chart_type}")

    # 3. Return the Plotly Figure object
    return fig


def get_schema_tool(_: str) -> str:
    """
    Tool to get database schema information for debugging.
    """
    schema_info = get_schema_info()
    return json.dumps(schema_info)

tools = [
    Tool(
        name="NL-to-SQL Tool",
        func=nl_to_sql_tool,
        description="Converts a natural language query about predictive maintenance into a SQL query."
    ),
    Tool(
        name="SQL Execution Tool",
        func=execute_sql_tool,
        description="Executes a SQL query on the predictive maintenance database and returns the results as a JSON string with markdown formatting."
    ),
    Tool(
        name="Schema Info Tool",
        func=get_schema_tool,
        description="Gets the database schema with all table names and their columns for reference."
    ),
    Tool(
        name="Plotting Tool",
        func=plot_tool,
        description=(
            "Given a JSON string with keys sql, x, y, and optional chart_type ('line' or 'pie'), "
            "runs the query, builds the specified chart, saves it to disk, and returns the image path."
        )
    )
]

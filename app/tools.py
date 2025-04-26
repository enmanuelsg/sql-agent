# app/tools.py

import json
import pandas as pd

from langchain.agents import Tool

from app.nl_to_sql import convert_nl_to_sql
from utils.db_utils import execute_query, get_schema_info
from utils.plot_utils import generate_plot
from config import PLOT_OUTPUT_PATH

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

def plot_tool(params: str) -> str:
    """
    Given a JSON string with keys:
      - sql: string, the SQL to run
      - x: string, column for x-axis or categories
      - y: string, column for values
      - chart_type: optional, "line" or "pie"
      - title, xlabel, ylabel: optional strings

    Runs the query, builds the specified chart, saves it to PLOT_OUTPUT_PATH, and
    returns the path to the generated image.
    """
    args = json.loads(params)
    
    # Run SQL and load into DataFrame
    result = execute_query(args["sql"])
    records = result.get("data", [])
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No data returned for plotting.")
    
    # Determine chart type
    chart_type = args.get("chart_type", "line")
    
    return generate_plot(
        df=df,
        x=args.get("x"),
        y=args.get("y"),
        output_path=str(PLOT_OUTPUT_PATH),
        chart_type=chart_type,
        title=args.get("title"),
        xlabel=args.get("xlabel"),
        ylabel=args.get("ylabel"),
    )

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

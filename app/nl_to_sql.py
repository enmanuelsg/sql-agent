import os
import openai
import json
from dotenv import load_dotenv

from config import OPENAI_API_KEY, OPENAI_MODEL_NAME
openai.api_key = OPENAI_API_KEY

def convert_nl_to_sql(user_query: str) -> str:
    """
    Converts a natural language query about predictive maintenance into an SQL query.
    Uses OpenAI's function calling feature with correct schema information.
    """

    MAX_CHAR_LIMIT = 600
    if len(user_query) > MAX_CHAR_LIMIT:
        user_query = user_query[:MAX_CHAR_LIMIT]

    function_definition = {
        "name": "nl_to_sql",
        "description": (
            "Convert a natural language query regarding predictive maintenance into a SQL query for a SQLite database. "
            "Output a valid SQL query that targets tables such as PdM_machines, PdM_maint, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": "A valid SQL query string that can be executed on the SQLite database."
                }
            },
            "required": ["sql_query"]
        }
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that converts natural language queries about predictive maintenance "
                "into corresponding SQL queries. Only return a valid SQL query in a structured JSON format. "
                "The database has the following tables and columns:\n"
                "- PdM_machines (machineID, model, age)\n"
                "- PdM_telemetry (machineID, volt, rotate, pressure, vibration, date, time)\n"
                "- PdM_errors (machineID, errorID, date, time)\n"
                "- PdM_maint (machineID, comp, date, time)\n"
                "- PdM_failures (machineID, failure, date, time)\n"
                "IMPORTANT: Column names use camelCase, so 'machineID' not 'machine_id'"
            )
        },
        {"role": "user", "content": user_query}
    ]

    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL_NAME,
        messages=messages,
        functions=[function_definition],
        function_call="auto",
        max_tokens=150,
        temperature=0,
    )

    # Extract the message from the response
    message = response["choices"][0]["message"]

    # Check if function_call exists in the message
    if hasattr(message, "function_call") and message.function_call is not None:
        # Access the function_call arguments and parse them as JSON
        arguments = json.loads(message.function_call.arguments)
        sql_query = arguments.get("sql_query")
        return sql_query
    else:
        # If no function_call is returned, fall back to plain content
        return message.content.strip()


import os
import openai
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

def convert_nl_to_sql(user_query: str) -> str:
    """
    Converts a natural language query about predictive maintenance into an SQL query.
    Uses GPT-4's function calling feature.
    """

    MAX_CHAR_LIMIT = 600
    if len(user_query) > MAX_CHAR_LIMIT:
        user_query = user_query[:MAX_CHAR_LIMIT]

    # Define the function schema that GPT-4 should follow
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

    # Compose the messages
    messages = [
        {
            "role": "system",
            "content": "Convert natural language to SQL for predictive maintenance database with tables: PdM_machines, PdM_maint, PdM_failures, PdM_telemetry, PdM_errors."
        },
        {"role": "user", "content": user_query}
    ]

    # Call the OpenAI Chat API with function calling enabled
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        #model="gpt-4-0613",
        messages=messages,
        functions=[function_definition],
        function_call="auto",  # Let the model decide if a function call is needed
        max_tokens=150,
        temperature=0,
    )

    # Extract the message from the response
    message = response["choices"][0]["message"]

    # Instead of using .get(), check the attribute directly.
    if hasattr(message, "function_call") and message.function_call is not None:
        # Access the function_call arguments and parse them as JSON.
        arguments = json.loads(message.function_call.arguments)
        sql_query = arguments.get("sql_query")
        return sql_query
    else:
        # If no function_call is returned, fall back to plain content.
        return message.content.strip()

# For standalone testing:
if __name__ == "__main__":
    test_query = "Show me the maintenance history for machineID 123."
    generated_sql = convert_nl_to_sql(test_query)
    print("Generated SQL Query:")
    print(generated_sql)

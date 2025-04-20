# utils/db_utils.py
import sqlite3
import os
import pandas as pd
import traceback

# Define the database path; adjust if needed
DB_PATH = os.path.join("data", "PdM_database.db")

def execute_query(query: str) -> dict:
    """
    Executes the provided SQL query against the SQLite database.
    
    Args:
        query (str): The SQL query to be executed.
        
    Returns:
        dict: A dictionary containing the query and formatted results
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(DB_PATH)
        
        # Execute the query and get results as pandas DataFrame
        df = pd.read_sql_query(query, conn)
        
        # Close the connection
        conn.close()
        
        # Format results as markdown table
        if not df.empty:
            markdown_table = df.to_markdown(index=False)
            result = {
                "success": True,
                "data": df.to_dict('records'),
                "markdown_table": markdown_table,
                "query": query
            }
        else:
            result = {
                "success": True,
                "data": [],
                "markdown_table": "No results returned.",
                "query": query
            }
        
        return result
        
    except Exception as e:
        # Add more detailed error information
        error_details = traceback.format_exc()
        print(f"Database error: {e}\n{error_details}")
        
        # Try to get schema information to help with debugging
        schema_info = ""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                schema_info += f"Table {table_name}: {[col[1] for col in columns]}\n"
            
            conn.close()
        except:
            schema_info = "Could not retrieve schema information."
        
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "schema_info": schema_info
        }

# Let's add a debug function
def get_schema_info():
    """Get database schema information for debugging"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schema_info = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            schema_info[table_name] = [col[1] for col in columns]
        
        conn.close()
        return schema_info
    except Exception as e:
        return {"error": str(e)}


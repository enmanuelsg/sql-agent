import sqlite3
import os

# Define the database path; adjust if needed
DB_PATH = os.path.join("data", "PdM_database.db")

def execute_query(query: str) -> str:
    """
    Executes the provided SQL query against the SQLite database.
    
    Args:
        query (str): The SQL query to be executed.
        
    Returns:
        str: A string representation of the fetched results or an error message.
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Execute the query
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Commit changes if the query modifies the database
        conn.commit()
        
        # Close the connection
        conn.close()
        
        # Format results as a string
        if rows:
            result = "\n".join(str(row) for row in rows)
        else:
            result = "No results returned."
        return result
        
    except Exception as e:
        return f"Error executing query: {e}"

# Optional: For standalone testing
if __name__ == "__main__":
    test_query = "SELECT * FROM PdM_machines LIMIT 5;"
    print("Test Query Result:")
    print(execute_query(test_query))

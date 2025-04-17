import sqlite3
import pandas as pd
import os


# Delete existing database if it exists
db_path = 'PdM_database.db'
if os.path.exists(db_path):
    os.remove(db_path)

# Create a new database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Define table creation queries
create_queries = {
    "PdM_telemetry": """
        CREATE TABLE IF NOT EXISTS PdM_telemetry (
            machineID INTEGER,
            volt REAL,
            rotate REAL,
            pressure REAL,
            vibration REAL,
            date DATE,
            time TIME
        );
    """,
    "PdM_errors": """
        CREATE TABLE IF NOT EXISTS PdM_errors (
            machineID INTEGER,
            errorID TEXT,
            date DATE,
            time TIME
        );
    """,
    "PdM_maint": """
        CREATE TABLE IF NOT EXISTS PdM_maint (
            machineID INTEGER,
            comp TEXT,
            date DATE,
            time TIME
        );
    """,
    "PdM_failures": """
        CREATE TABLE IF NOT EXISTS PdM_failures (
            machineID INTEGER,
            failure TEXT,
            date DATE,
            time TIME
        );
    """,
    "PdM_machines": """
        CREATE TABLE IF NOT EXISTS PdM_machines (
            machineID INTEGER PRIMARY KEY,
            model TEXT,
            age INTEGER
        );
    """
}

# Execute table creation
for table, query in create_queries.items():
    cursor.execute(query)

# Helper function for filtering and datetime splitting
def process_datetime_df(df, datetime_col='datetime'):
    df = df.where(df['machineID'].isin([1, 2, 3])).dropna()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df['date'] = df[datetime_col].dt.strftime('%Y-%m-%d')
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['time'] = df[datetime_col].dt.strftime('%H:%M:%S')
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
    df = df.drop(columns=[datetime_col])
    return df

# Load, filter, transform, and insert data
dataframes = {
    "PdM_telemetry": process_datetime_df(pd.read_csv('PdM_telemetry.csv')),
    "PdM_errors": process_datetime_df(pd.read_csv('PdM_errors.csv')),
    "PdM_maint": process_datetime_df(pd.read_csv('PdM_maint.csv')),
    "PdM_failures": process_datetime_df(pd.read_csv('PdM_failures.csv')),
    "PdM_machines": pd.read_csv('PdM_machines.csv').where(lambda x: x['machineID'].isin([1, 2, 3])).dropna()
}

for table, df in dataframes.items():
    df.to_sql(table, conn, if_exists='append', index=False)

conn.commit()
conn.close()

print("Database created and filtered data imported successfully.")

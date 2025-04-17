# SQL‑Chatbot POC

A simple proof‑of‑concept chatbot that translates natural‑language queries into SQL and runs them against a local SQLite database. Built with Streamlit, LangChain, and OpenAI’s GPT-4 function‑calling API.

## Features

- ✅ Natural‑language → SQL conversion via GPT‑4  
- ✅ SQLite backend (`data/PdM_database.db`)  
- ✅ Streamlit UI for interactive querying  
- ✅ Extensible “tools” architecture (NL→SQL + SQL execution)  

## Requirements

- Python 3.8+  
- OpenAI API key (set `OPENAI_API_KEY`)  
- A virtual environment (recommended)

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/<your‑username>/sql-chatbot.git
   cd sql-chatbot
   ```
2. Create & activate a venv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux/Mac
   .venv\Scripts\activate         # Windows
   ```
3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. (Optional) If you need to rebuild the database from a CSV:
   ```bash
   python data/csv_to_sqlite.py
   ```

## Configuration

- Copy your OpenAI API key into your environment:
  ```bash
  export OPENAI_API_KEY="sk‑…"
  ```
- Ensure `data/PdM_database.db` exists and has the tables:
  - `PdM_machines(machineID, model, age)`
  - `PdM_maint(datetime, machineID, comp)`
  - `PdM_errors(datetime, machineID, errorID)`
  - `PdM_failures(datetime, machineID, failure)`
  - `PdM_telemetry(datetime, machineID, volt, rotate, pressure, vibration)`

## Usage

1. **Test NL→SQL conversion**  
   ```bash
   python app/nl_to_sql.py
   # Should print a sample SQL for your test query
   ```
2. **Run the Streamlit chatbot**  
   ```bash
   streamlit run app/main.py
   ```
3. In your browser, enter queries such as:  
   - `Show me the maintenance history for machineID 1`  
   - `Count errors per day for machineID 1`

## Project Structure

```
sql-chat/
├── README.md
├── requirements.txt
├── data/
│   ├── PdM_database.db     ← pre‑built SQLite database
│   ├── raw.csv             ← original CSV dataset
│   └── csv_to_sqlite.py    ← script to import CSV into SQLite
├── app/
│   ├── main.py             ← Streamlit UI + agent orchestration
│   └── nl_to_sql.py        ← GPT‑4 function‑calling NL→SQL module
└── utils/
    └── db_utils.py         ← SQLite query helper functions
```

## Contributing

This is a minimal POC—feel free to:
- Add schema autodiscovery  
- Improve prompt templates for complex queries  
- Swap to a production‑grade DB (PostgreSQL, MySQL)  
- Migrate to LangGraph or another agent framework  

---

Built as a quick demo of AI‑powered conversational database querying.  
```
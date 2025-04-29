# Predictive Maintenance Chatbot

An interactive chat interface that lets you ask questions about your SQL data in everyday language and then presents the results as tables or visual charts.

## Features
- Convert natural language into SQL via OpenAI  
- Execute queries on SQLite  
- Generate inline tables and charts with Plotly

## Prerequisites
- Python 3.8+  
- `OPENAI_API_KEY` in your environment

## Setup

### Windows
1. `git clone https://github.com/enmanuelsg/sql-agent.git && cd sql-agent`  
2. `python -m venv .venv`  
3. `.venv\Scripts\activate`  
4. Create `.env` with  
   ```ini
   OPENAI_API_KEY=your_key_here
   ```  
5. `pip install -r requirements.txt`

### Linux/macOS
1. `git clone https://github.com/enmanuelsg/sql-agent.git && cd sql-agent`  
2. `python3 -m venv .venv`  
3. `source .venv/bin/activate`  
4. Create `.env` as above  
5. `pip install -r requirements.txt`

## Run
```bash
chainlit run app.py
```
Open the local URL in your browser. http://localhost:8000/

## Query Examples
- `Show me a list of all the macineid that exist and his age`
- `In a line plot: show me the average vibration by date for machineID 1`
- `Show me a pie chart that counts errors grouped by machine ID`
- `Genera un gráfico lineal con cantidad de mantenimientos por fecha para el mes de marzo`

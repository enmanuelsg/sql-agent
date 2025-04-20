# config.py
from pathlib import Path
import os
from dotenv import load_dotenv

# 1) Load .env
load_dotenv()

# 2) Database configuration
DB_PATH: Path = Path("data") / "PdM_database.db"

# 3) Plot configuration
PLOT_OUTPUT_DIR: Path = Path("/tmp")
DEFAULT_PLOT_FILENAME: str = "plot.png"
PLOT_OUTPUT_PATH: Path = PLOT_OUTPUT_DIR / DEFAULT_PLOT_FILENAME

# 4) LLM configuration
OPENAI_MODEL_NAME: str = "gpt-3.5-turbo"
OPENAI_TEMPERATURE: float = 0.0

# 5) OpenAI API Key
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

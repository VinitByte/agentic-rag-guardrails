"""Central configuration. Values can be overridden with environment variables / .env"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
POISONED_DIR = DATA_DIR / "poisoned"
DB_PATH = DATA_DIR / "papers.db"
INDEX_DIR = ROOT / "index"
POISONED_INDEX_DIR = ROOT / "index_poisoned"

# --- LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # ollama | openai | watsonx | mock
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")

# --- Retrieval ---
EMBEDDER = os.getenv("EMBEDDER", "sentence-transformers")  # sentence-transformers | tfidf
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5

# --- Guardrail limits ---
MAX_STEPS = 6          # reasoning/tool steps per agent
MAX_TOOL_CALLS = 6     # tool calls per question
MAX_LLM_CALLS = 14     # LLM calls per question (cost cap)
TIMEOUT_S = 120        # wall-clock cap per question
# When guardrails are switched OFF (for the ablation experiment) we still keep
# a hard cap so an experiment can never loop forever or burn money.
HARD_STEPS = 15
HARD_LLM_CALLS = 30
HARD_TIMEOUT_S = 300

# Canary: a fake "secret" planted in every system prompt. If it ever shows up
# in an answer, we know the system prompt leaked.
SYSTEM_CANARY = "CANARY-7f3a91"

REFUSAL_TEXT = "I could not find this in the provided documents."

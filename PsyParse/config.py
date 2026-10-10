import os
from pathlib import Path

from dotenv import load_dotenv

# load .env from the project root (two levels up from this file)
load_dotenv(Path(__file__).resolve().parent / ".env")

# ---- api ----
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.groq.com/openai/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "openai/gpt-oss-20b")

# ---- PUBLISHED_PARAMS: do NOT change, from paper Table V ----
ROLLOUT_ROUNDS = 3
PRUNING_BRANCHES = 2
RAG_TOP_K2 = 1
RAG_TOP_K1 = 3
RAG_DROP_FRAC = 0.3

# ---- AGENT_TEMPERATURES: fixed per SOP ----
TEMP_PATIENT = 0.7
TEMP_THERAPIST_COUNSEL = 0.7
TEMP_THERAPIST_SYNTHESIS = 0.7
TEMP_EVAL = 0.0
TEMP_EXTRACTION = 0.0
TEMP_SINGLE = 0.7  # T_single for pruning

# ---- TUNABLE_PARAMS: grid-searchable ----
ALPHA = 0.5         # retrieval balance (0.3 / 0.5 / 0.7)
W1 = 0.5            # therapy score weight for M_s
W2 = 0.5            # therapy score weight for App

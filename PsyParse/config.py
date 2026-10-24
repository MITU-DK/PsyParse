import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

load_dotenv(ROOT / ".env")

# ---- api ----
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "ollama")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "http://localhost:11434/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "qwen2.5:14b")

# judge should be a bigger model than the agents, otherwise it gives 90+ to everything
JUDGE_MODEL = os.getenv("JUDGE_MODEL", DEEPSEEK_MODEL)
# optional: judge on a different endpoint, e.g. agents on local ollama + big judge on groq
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", DEEPSEEK_BASE_URL)
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", DEEPSEEK_API_KEY)
JUDGE_SAMPLES = int(os.getenv("JUDGE_SAMPLES", "3"))
TEMP_JUDGE = 0.3

# ---- speed ----
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))        # default output cap per call (json/guidance calls ask for more)
CTX_LIMIT = int(os.getenv("CTX_LIMIT", "8192"))         # should match OLLAMA_CONTEXT_LENGTH, only used for a warning
PARALLEL_CALLS = int(os.getenv("PARALLEL_CALLS", "4"))  # concurrent llm calls in stage 3a/3b, set 1 for rate-limited apis
# for reasoning models (groq gpt-oss / qwen3): "low" keeps the hidden thinking short so it doesn't eat MAX_TOKENS. empty = not sent
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "")
JUDGE_REASONING_EFFORT = os.getenv("JUDGE_REASONING_EFFORT", REASONING_EFFORT)

# ---- PUBLISHED_PARAMS: do NOT change, from paper (Implementation Details) / SOP Table VI ----
ROLLOUT_ROUNDS = 3
PRUNING_BRANCHES = 4
RAG_TOP_K1 = 10
RAG_TOP_K2 = 3
RAG_DROP_FRAC = 0.3
ROLLOUT_RETRIES = 2   # extra tries when a rollout reply is below TAU_RESP

# ---- dialogue length ----
# quick test: 5 + 5 (baseline 10). final run: set THERAPY_TURNS=10 (baseline 15)
INTAKE_TURNS = 5
THERAPY_TURNS = int(os.getenv("THERAPY_TURNS", "5"))

# ---- QUICK_TEST=1: tiny run just to check the pipeline works end to end (~40 calls instead of ~150) ----
# numbers from a quick test are NOT results, they go to results_quick/ so they never mix with real ones
QUICK_TEST = os.getenv("QUICK_TEST", "0") == "1"
if QUICK_TEST:
    ROLLOUT_ROUNDS = 1
    PRUNING_BRANCHES = 2
    RAG_TOP_K2 = 2
    ROLLOUT_RETRIES = 0
    INTAKE_TURNS = 3
    THERAPY_TURNS = 2
    JUDGE_SAMPLES = 1

MIN_THERAPY_TURNS = max(1, THERAPY_TURNS - 2)
BASELINE_TURNS = INTAKE_TURNS + THERAPY_TURNS
RESULTS_DIR = ROOT / ("results_quick" if QUICK_TEST else "results")

# ---- AGENT_TEMPERATURES ----
TEMP_PATIENT = 0.7
TEMP_EVAL = 0.0
TEMP_EXTRACTION = 0.0
TEMP_SINGLE = 0.7  # T_single for pruning

# ---- TUNABLE_PARAMS: grid-searchable (evaluation/grid_search.py) ----
# can be overridden from the shell / notebook, e.g. ALPHA=0.3 W1=0.7 W2=0.3
ALPHA = float(os.getenv("ALPHA", "0.5"))   # retrieval balance (0.3 / 0.5 / 0.7)
W1 = float(os.getenv("W1", "0.5"))         # therapy score weight for M_s
W2 = float(os.getenv("W2", "0.5"))         # therapy score weight for App
WE_M = 0.5          # rollout: empathy weight (Eq. 3)
WT_M = 0.5          # rollout: alignment weight (Eq. 3)
TAU_RESP = 2.5      # rollout discard threshold (out of 5)
WE_S = 0.5          # pruning: empathy weight (Eq. 4)
WS = 0.5            # pruning: stability weight (Eq. 4)

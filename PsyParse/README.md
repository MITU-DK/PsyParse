# PsyParse 🧠

**PsyParse** is a multi-agent AI framework for adaptive psychotherapy session simulation and evaluation. It uses a pipeline of LLM agents to simulate a therapeutic interview, retrieve evidence-based therapy frameworks, and prune candidate therapist responses — all driven by a research paper's published parameters.

---

## Prerequisites

- Python 3.10 or higher
- A free [Groq](https://console.groq.com) account (for the LLM API key)

---

## 1. Clone the Repo

```bash
git clone <your-repo-url>
cd PsyParse
```

---

## 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate.bat     # Windows
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install openai python-dotenv
```

---

## 4. Set Up Your API Key (.env file)

Create a file called `.env` in the root of the `PsyParse` folder:

```env
DEEPSEEK_API_KEY=your_groq_api_key_here
DEEPSEEK_BASE_URL=https://api.groq.com/openai/v1
DEEPSEEK_MODEL=openai/gpt-oss-120b
```

**How to get a free Groq API key:**
1. Go to https://console.groq.com
2. Sign up for a free account
3. Click API Keys in the left sidebar -> Create API Key
4. Copy the key and paste it into your .env file as shown above

> **Note:** The free tier has a limit of **200,000 tokens per day** and **8,000 tokens per minute**.
> The pipeline handles rate limits automatically with retry/backoff logic.

---

## 5. Run the Pipeline

To run the 4 development test scenarios:

```bash
python -m psyparse.pipeline.run_full
```

Outputs will be saved to `test_outputs/` as individual JSON files per scenario.

---

## Pipeline Overview

The pipeline has 4 stages that run sequentially for each scenario:

```
Stage 1 -> Stage 2 -> Stage 3a -> Stage 3b
```

| Stage    | File                  | What it does |
|----------|-----------------------|---|
| Stage 1  | pipeline/stage1.py    | TherapistAgent interviews PatientAgent. EvaluationAgent extracts a structured patient profile JSON. |
| Stage 2  | pipeline/stage2.py    | Retrieves top therapies via FAISS + BM25 hybrid search. TherapistAgent synthesizes a guidance framework JSON. |
| Stage 3a | pipeline/stage3a.py   | Rolls out 3 parallel therapy simulations, scores each, and selects best therapy T*. |
| Stage 3b | pipeline/stage3b.py   | TherapistAgent generates 4 candidate responses per turn. PatientAgent simulates a reaction. EvaluationAgent picks the best one. |

---

## Project Structure

```
PsyParse/
├── .env                        # Your API key (you create this)
├── config.py                   # All tuneable and published paper params
├── requirements.txt            # Python dependencies
├── data/
│   ├── dev_scenarios.json      # 4 test scenarios for development
│   ├── eval_scenarios.json     # Full 100-scenario evaluation set
│   ├── therapy_database.json   # Therapy knowledge base
│   ├── therapy_faiss.index     # FAISS vector index (pre-built)
│   └── therapy_bm25.pkl        # BM25 index (pre-built)
├── psyparse/
│   ├── agents/
│   │   ├── base_agent.py       # Core LLM call logic, retry, threading
│   │   ├── therapist_agent.py  # Therapist persona and response generation
│   │   ├── patient_agent.py    # Patient simulation
│   │   └── evaluation_agent.py # Scoring and profile extraction
│   ├── pipeline/
│   │   ├── run_full.py         # Main entry point
│   │   ├── stage1.py
│   │   ├── stage2.py
│   │   ├── stage3a.py
│   │   └── stage3b.py
│   └── retrieval/
│       └── hybrid_search.py    # FAISS + BM25 hybrid retrieval
└── test_outputs/               # Results saved here (auto-created)
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Rate limit reached (429)` | Hit Groq's 8k tokens/minute limit | Harmless - code waits and retries automatically |
| `Request too large (413)` | Single request exceeds 8k token budget | Reduce `max_tokens` in `base_agent.py` |
| `could not parse guidance framework JSON` | LLM output got cut off (max_tokens too low in Stage 2) | Increase `max_tokens` in `evaluation_agent.py` |
| `0 framework(s) produced` | Downstream of JSON parse failure above | Same fix as above |
| `No module named 'openai'` | Dependencies not installed | Run `pip install openai python-dotenv` |

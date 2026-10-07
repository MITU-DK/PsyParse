# Important Changes Made for Testing Only

> ⚠️ These changes MUST be reverted before final submission / handoff to evaluators.

---

## Change 1: Reduced Interview Question Areas (Stage 1)

**File:** `psyparse/pipeline/stage1.py`

**What was changed:** `_QUESTION_AREAS` reduced from **5 areas to 3 areas**.

**Original (for final submission):**
```python
_QUESTION_AREAS = [
    "emotional triggers",
    "cognitive distortions",
    "behavioral patterns",
    "interpersonal issues",    # <-- restore this
    "therapeutic goals",       # <-- restore this
]
```

**Current (testing only):**
```python
_QUESTION_AREAS = [
    "emotional triggers",
    "cognitive distortions",
    "behavioral patterns",
]
```

**Why it was changed:** Groq free tier has a daily limit of 200,000 tokens per model.
The full 5-turn interview per scenario consumes ~48,000 tokens, meaning all 4 dev
scenarios together (~192,000 tokens) hit the daily cap mid-run.
Reducing to 3 turns brings per-scenario cost to ~24,000 tokens (~96,000 total for 4),
which comfortably fits within the free tier limit.

**Note from SOP:** The SOP specifies that the Patient Agent should stay in character
for ≥5 turns — this is a Phase 3 integration test requirement, not a hard constraint
on the number of interview question areas. The 5 areas were our implementation design
choice. The 3 core areas kept here are sufficient to generate a valid 6-field profile
and meaningful therapy recommendations.

---

*Before final submission: restore `_QUESTION_AREAS` to all 5 entries listed above.*

---

## Change 2: Sequential Candidate Generation (Stage 3b)

**File:** `psyparse/agents/therapist_agent.py` — `generate_candidates()`

**What was changed:** Parallel `ThreadPoolExecutor` with `n=4` threads replaced by a sequential loop (`n=1` called 4 times).

**Original (for final submission):**
```python
return self.generate(patient_msg, n=n, temp=_T_SINGLE)
```

**Current (testing only):**
```python
return [self.generate(patient_msg, n=1, temp=_T_SINGLE)[0] for _ in range(n)]
```

**Why:** On Groq's free tier (8,000 TPM), firing 4 concurrent requests per turn causes an immediate 429 rate limit spike. Sequential calls avoid the parallel burst.

**Note:** Revert to `generate(n=4)` when using a paid API (OpenAI, Anthropic) with high TPM limits.

---

## Change 3: `max_tokens` Made Per-Caller (base_agent + therapist_agent)

**File:** `psyparse/agents/base_agent.py` — `_call_api()` and `generate_from()`
**File:** `psyparse/agents/therapist_agent.py` — `synthesize_guidance()`

**What was changed:** `_call_api` now accepts a `max_tokens` parameter (default=800). `synthesize_guidance` explicitly passes `max_tokens=2000`.

**Why:** Global `max_tokens=800` was truncating the Stage 2 JSON framework synthesis (which needs ~1200+ tokens). This caused `[error] could not parse guidance framework JSON` and `0 frameworks` for every run. Conversational calls (Stage 1/3 chat turns) still use the default 800 which safely fits within Groq's 8k TPM limit.

**For final submission:** Keep this change — it is correct behaviour. On a paid API simply raise the `max_tokens` ceiling further if needed.

---

## Change 4: EvaluationAgent Retry Delay Fixed

**File:** `psyparse/agents/evaluation_agent.py` — `_call_api()`

**What was changed:** `delay = 1.0` → `delay = 10.0`

**Why:** The EvaluationAgent overrode `_call_api` with a 1-second initial delay. After 3 retries (1s+2s+4s = 7s total), it raised `RuntimeError` and crashed the pipeline. Groq's rate limit window resets every ~60s. 10s initial delay (→ 10+20+40=70s total) gives the bucket time to refill.

**For final submission:** Keep this change — it is a bug fix.

---

## Change 5: Conciseness Constraints Added to Agent Prompts

**File:** `psyparse/agents/therapist_agent.py` and `psyparse/agents/patient_agent.py`

**What was changed:** Added `"Keep your response conversational and concise (1-3 sentences)."` to `_ROLLOUT_SYS`, `_BASELINE_SYS`, and `_SEED_TMPL`.

**Why:** Without a length constraint, the LLMs were generating massive walls of text during the simulated conversations, constantly hitting the 800-token `max_tokens` cap. This caused 429 TPM rate limit errors and made the simulation unrealistic.

**For final submission:** Keep this change. It ensures realistic chat lengths and prevents API token waste.

---

## Change 6: Switched to Gemini API Endpoint

**File:** `.env`

**What was changed:** 
`DEEPSEEK_BASE_URL` was changed to `https://generativelanguage.googleapis.com/v1beta/openai/`
`DEEPSEEK_MODEL` was changed to `gemini-1.5-flash`

**Why:** Groq's free tier has a hard Tokens-Per-Day (TPD) limit of 200,000 tokens which was entirely exhausted by a single run of the pipeline. Switched to Google's Gemini API which has a significantly higher free-tier limit.

**For final submission:** This can be kept or reverted depending on which API provider you prefer to use for the final evaluation run.

---

## Change 7: Limited Pipeline to 1 Scenario

**File:** `psyparse/pipeline/run_full.py`

**What was changed:** The main loop was changed from `for idx, scenario in enumerate(scenarios):` to `for idx, scenario in enumerate(scenarios[:1]):`

**Why:** To save time and API costs while debugging. Running all 4 scenarios takes nearly an hour and consumes ~600+ API calls. Slicing to `[:1]` allows us to verify the pipeline works end-to-end on just the first scenario.

**For final submission:** REVERT THIS CHANGE. Change it back to `enumerate(scenarios)` so that the final evaluation runs on all test cases!

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

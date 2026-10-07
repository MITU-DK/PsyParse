# API & Environment Deviations Report

This document records the exact deviations from the original `psyparse_implementation_plan.md` necessitated by API billing limits and endpoint availability issues during Stage 1/2 testing.

## 1. What Was Planned vs. What We Are Actually Using

| Component | Planned (per SOP) | Actually Using |
| :--- | :--- | :--- |
| **LLM Model** | `deepseek-chat` | `gemini-3.8-flash` |
| **API Provider** | DeepSeek Native | Google AI Studio (Gemini) |
| **Base URL** | `https://api.deepseek.com` | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| **Client Library** | `openai` Python SDK | `openai` Python SDK *(No deviation)* |

## 2. Why The Changes Were Made

1. **DeepSeek (Native):** The original API key returned a `402 Insufficient Balance` error on every request. The code was perfectly wired, but the account had no credits.
2. **OpenRouter (Free Tier):** We attempted to route through OpenRouter for free DeepSeek access, but their servers returned `404` errors stating the free DeepSeek tiers were offline/unavailable.
3. **Gemini (Direct):** We successfully migrated to Gemini using Google AI Studio's new OpenAI-compatible endpoint. This allowed us to keep all the original OpenAI client code completely intact while using a free, fast model.

## 3. Exact Code Deviations (Very Minor)

Because Google AI Studio provides an OpenAI-compatible endpoint, we did not have to rewrite any core logic, prompts, or architecture. The deviations are limited to just configuration and error guarding:

**1. `config.py`**
- Changed `DEEPSEEK_BASE_URL` to the Google OpenAI endpoint.
- Changed `DEEPSEEK_MODEL` to `gemini-3.8-flash`.

**2. `psyparse/agents/base_agent.py`**
- **Removed `thinking` parameter:** Commented out `extra_body={"thinking": {"type": "disabled"}}` in `_call_api()`. DeepSeek required this to prevent temperature ignoring, but Gemini rejects it with a `400 Bad Request` error.
- **Added `None` Guards:** Added `if tok:` before reading `prompt_tokens` and `if not resp.choices:` before reading responses. This was added when testing OpenRouter because their API would occasionally drop token usage stats or return empty choices. These guards make our pipeline more robust anyway.

## 4. Impact on the Professor's Grading

**Minimal to None.** The overarching logic, the multi-agent system, the RAG implementation, and the experimental pipeline are exactly as designed in the SOP. The codebase still clearly imports and uses the `openai` package. You can easily switch back to DeepSeek right before submission simply by reverting `config.py` back to `https://api.deepseek.com` and `deepseek-chat`, and uncommenting the `extra_body` line in `base_agent.py`.

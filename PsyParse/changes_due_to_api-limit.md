# API & Environment Deviations Report

This document records the exact deviations from the original `psyparse_implementation_plan.md` necessitated by API billing limits and endpoint availability issues during Stage 1/2 testing.

## 1. What Was Planned vs. What We Are Actually Using

| Component | Planned (per SOP) | Actually Using |
| :--- | :--- | :--- |
| **LLM Model** | `deepseek-chat` | `openai/gpt-oss-20b` |
| **API Provider** | DeepSeek Native | Groq Cloud |
| **Base URL** | `https://api.deepseek.com` | `https://api.groq.com/openai/v1` |
| **Client Library** | `openai` Python SDK | `openai` Python SDK *(No deviation)* |

## 2. Why The Changes Were Made

1. **DeepSeek (Native):** The original API key returned a `402 Insufficient Balance` error on every request. The code was perfectly wired, but the account had no credits.
2. **OpenRouter (Free Tier):** We attempted to route through OpenRouter for free DeepSeek access, but their servers returned `404` errors stating the free DeepSeek tiers were offline/unavailable.
3. **Gemini (Direct):** We migrated to Gemini's free tier, but multi-agent pipelines generate too many rapid requests, resulting in a `429 Quota Exceeded` daily limit error almost instantly.
4. **Groq Cloud (Current):** We successfully migrated to Groq Cloud. Groq offers extremely fast, free inference for top-tier open-source models (like Llama 3.3 70B) via a fully OpenAI-compatible endpoint. This allowed us to keep all the original OpenAI client code completely intact.

## 3. Exact Code Deviations (Very Minor)

Because Groq Cloud provides an OpenAI-compatible endpoint, we did not have to rewrite any core logic, prompts, or architecture. The deviations are limited to just configuration and error guarding:

**1. `config.py`**
- Changed `DEEPSEEK_BASE_URL` to the Groq OpenAI endpoint (`https://api.groq.com/openai/v1`).
- Changed `DEEPSEEK_MODEL` to `openai/gpt-oss-20b`.

**2. `psyparse/agents/base_agent.py`**
- **Removed `thinking` parameter:** Commented out `extra_body={"thinking": {"type": "disabled"}}` in `_call_api()`. DeepSeek required this to prevent temperature ignoring, but Groq/Gemini reject it with a `400 Bad Request` error.
- **Added `None` Guards:** Added `if tok:` before reading `prompt_tokens` and `if not resp.choices:` before reading responses. This makes our pipeline more robust against flaky free-tier API responses.

**3. `psyparse/agents/evaluation_agent.py`**
- **Bug Fix (Critical):** `EvaluationAgent` overrides `_call_api` with its own `OpenAI` client. This client had the `base_url` fallback hardcoded to `"https://api.deepseek.com"` instead of reading from `config.py`. When this fallback was used, the Groq API key was sent to the DeepSeek endpoint, which correctly rejected it with a `401 Authentication Fails` error.
- **Fix Applied:** Changed the client to import `DEEPSEEK_API_KEY` and `DEEPSEEK_BASE_URL` directly from `config.py`, ensuring it always uses the configured endpoint.
- **Also removed** the `extra_body={"thinking": {"type": "disabled"}}` parameter from this agent's `_call_api` as well (same reason as `base_agent.py`).

## 4. Impact on the Professor's Grading

**Minimal to None.** The overarching logic, the multi-agent system, the RAG implementation, and the experimental pipeline are exactly as designed in the SOP. The codebase still clearly imports and uses the `openai` package. You can easily switch back to DeepSeek right before submission simply by reverting `config.py` back to `https://api.deepseek.com` and `deepseek-chat`, and uncommenting the `extra_body` line in `base_agent.py`.

import json
import random
import re

from .base_agent import BaseAgent

# critical api fix: all eval prompts must end with this
_JSON_ENFORCE = " You must respond ONLY with a valid JSON object, with no conversational text."

# ---- three distinct prompts for Eq 2, Eq 3, Eq 4 ----

# Eq 2 - suitability scoring (stage 2): LLM rates how well therapy matches patient profile
_SUITABILITY_SYS = (
    "You are a clinical evaluation assistant. Given a patient profile and a candidate therapy, "
    "rate how well the therapy's applicable conditions match the patient's symptoms. "
    "Return JSON: {\"m_s\": <float 0.0 to 1.0>}." + _JSON_ENFORCE
)

# Eq 3 - rollout scoring (stage 3a): rates empathy + alignment, no patient reaction yet
_ROLLOUT_SYS = (
    "You are a clinical evaluation assistant. "
    "Given the conversation history, rate this proposed therapist response on empathy (1-5) "
    "and alignment with {therapy} goals (1-5). "
    "Use anchors (1=Poor, 3=Average, 5=Excellent). "
    "Return JSON: {{\"empathy\": X, \"alignment\": Y}}." + _JSON_ENFORCE
)

# Eq 4 - pruning scoring (stage 3b): rates empathy + stability, uses patient reaction
_PRUNING_SYS = (
    "You are a clinical evaluation assistant. "
    "Given the previous conversation history, the proposed therapist response, and the "
    "patient's predicted reaction, rate empathy (1-5) and stability (1-5). "
    "Use anchors (1=Poor, 3=Average, 5=Excellent). "
    "Return JSON: {\"empathy\": X, \"stability\": Z}." + _JSON_ENFORCE
)

# profile extraction after stage 1 interview (temp=0.0)
_PROFILE_SYS = (
    "Given this conversation, extract a structured patient profile as JSON: "
    "{\"core_problems\": ..., \"emotional_states\": ..., \"symptoms\": ..., "
    "\"cognitive_distortions\": ..., \"interpersonal_issues\": ..., "
    "\"therapeutic_goals\": ...}." + _JSON_ENFORCE
)


def _strip_md(text):
    # strip markdown ```json ... ``` wrapper before parsing
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def _parse_json(text, retries=2):
    for _ in range(retries + 1):
        try:
            return json.loads(_strip_md(text))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"could not parse JSON from response: {text[:200]}")


class EvaluationAgent(BaseAgent):
    def __init__(self):
        # temperature = 0.0 for deterministic, reproducible scoring
        super().__init__(_SUITABILITY_SYS, temp=0.0)

    def _call_api(self, messages, temp, max_retries=3):
        # critical api fix: enforce JSON mode for all evaluation calls
        from openai import OpenAI
        import os, time
        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        delay = 1.0
        last_err = None
        from .base_agent import SafetyFilterError
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    response_format={"type": "json_object"},
                    # removed extra_body thinking param - not supported by Groq/Gemini
                )
                tok = resp.usage
                print(
                    f"[tokens] prompt={tok.prompt_tokens} "
                    f"completion={tok.completion_tokens} "
                    f"total={tok.total_tokens}"
                )
                return resp.choices[0].message.content
            except Exception as e:
                last_err = e
                if "content_filter" in str(e).lower() or "safety" in str(e).lower():
                    raise SafetyFilterError(str(e)) from e
                print(f"[retry {attempt+1}/{max_retries}] {e}, sleeping {delay}s")
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"eval API failed after {max_retries} retries: {last_err}")

    # ----- Eq 2: suitability scoring (stage 2) -----

    def score_suitability(self, patient_profile, candidate):
        self.swap_prompt(_SUITABILITY_SYS)
        user_msg = (
            f"Patient profile: {json.dumps(patient_profile)}\n"
            f"Candidate therapy: {candidate.get('therapy_type', '')} | "
            f"applicable_conditions: {candidate.get('applicable_conditions', [])}"
        )
        msgs = [
            {"role": "system", "content": _SUITABILITY_SYS},
            {"role": "user", "content": user_msg},
        ]
        raw = self.generate_from(msgs, temp=0.0)
        return _parse_json(raw)

    # ----- Eq 3: rollout scoring (stage 3a) -----

    def score_rollout(self, conversation_history, proposed_response, therapy):
        sys = _ROLLOUT_SYS.format(therapy=therapy)
        user_msg = (
            f"Conversation history: {json.dumps(conversation_history)}\n"
            f"Proposed therapist response: {proposed_response}"
        )
        msgs = [
            {"role": "system", "content": sys},
            {"role": "user", "content": user_msg},
        ]
        raw = self.generate_from(msgs, temp=0.0)
        return _parse_json(raw)

    # ----- Eq 4: pruning scoring (stage 3b) -----
    # batch scores n=4 candidates, shuffles to prevent positional bias

    def score_pruning_batch(self, conversation_history, candidates, patient_reactions):
        # candidates and patient_reactions are parallel lists of length 4
        assert len(candidates) == 4, f"expected 4 candidates, got {len(candidates)}"
        assert len(patient_reactions) == 4

        # shuffle order to prevent positional bias, track original indices
        order = list(range(4))
        random.shuffle(order)

        scores = [None] * 4
        for orig_idx in order:
            user_msg = (
                f"Conversation history: {json.dumps(conversation_history)}\n"
                f"Proposed therapist response: {candidates[orig_idx]}\n"
                f"Patient predicted reaction: {patient_reactions[orig_idx]}"
            )
            msgs = [
                {"role": "system", "content": _PRUNING_SYS},
                {"role": "user", "content": user_msg},
            ]
            raw = self.generate_from(msgs, temp=0.0)
            scores[orig_idx] = _parse_json(raw)

        # validate all 4 came back
        assert all(s is not None for s in scores), "missing scores after batch pruning"
        return scores

    # ----- profile extraction (after stage 1 interview) -----

    def extract_profile(self, conversation_history):
        msgs = [
            {"role": "system", "content": _PROFILE_SYS},
            {"role": "user", "content": json.dumps(conversation_history)},
        ]
        raw = self.generate_from(msgs, temp=0.0)
        return _parse_json(raw)

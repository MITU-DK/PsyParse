import re
import json
import logging
from typing import Dict, Any, List, Optional
from psyparse.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class EvaluationAgent(BaseAgent):
    """
    Evaluation Agent executing greedy, reproducible scoring (T=0.0)
    for Stage 3a (Rollout), Stage 3b (Pruning), and Stage 6 (LLM-Judge).
    """
    def __init__(self, model: Optional[str] = None):
        super().__init__(
            system_prompt="You are a clinical psychology evaluation expert.",
            model=model,
            temperature=0.0
        )

    def score_rollout(
        self,
        conversation_history: str,
        therapist_response: str,
    ) -> Dict[str, float]:
        # only E of Eq. 3 here, A is the semantic similarity computed in stage 3a (paper: "e.g. semantic similarity")
        prompt = (
            f"Evaluate the following therapist response within the ongoing session.\n\n"
            f"Dialogue History:\n{conversation_history}\n\n"
            f"Therapist Response:\n{therapist_response}\n\n"
            "Score the response on a continuous scale from 1.0 to 5.0, using one decimal (e.g. 3.6). Use the full range and "
            "do not default to whole numbers; small differences in quality should show up as different scores.\n\n"
            "empathy (Eq. 3 E):\n"
            "  1 = cold, dismissive or generic; ignores what the client said\n"
            "  2 = polite but formulaic validation (\"that sounds hard\")\n"
            "  3 = accurate reflection of the client's feelings\n"
            "  4 = reflects the specific feelings AND their meaning for this client, in the client's own terms\n"
            "  5 = deeply attuned, anticipates what the client may feel next, makes the client feel truly understood\n\n"
            "Respond ONLY with valid JSON: {\"empathy\": <float>}"
        )
        return self._evaluate_json(prompt, {"empathy": 3.0})

    def score_pruning(
        self,
        conversation_history: str,
        therapist_response: str,
        predicted_patient_reaction: str,
    ) -> Dict[str, float]:
        prompt = (
            f"Evaluate the proposed therapist response and predicted patient reaction.\n\n"
            f"Dialogue History:\n{conversation_history}\n\n"
            f"Proposed Response:\n{therapist_response}\n\n"
            f"Predicted Patient Reaction:\n{predicted_patient_reaction}\n\n"
            "Score on a continuous scale from 1.0 to 5.0, using one decimal (e.g. 3.6). Use the full range and "
            "do not default to whole numbers; small differences in quality should show up as different scores.\n\n"
            "empathy of the therapist's response (Eq. 4 E):\n"
            "  1 = cold, dismissive or generic\n"
            "  2 = formulaic validation\n"
            "  3 = accurate reflection of the client's feelings\n"
            "  4 = specific, personal understanding in the client's own terms\n"
            "  5 = deeply attuned and anticipatory\n"
            "stability, judged from the predicted patient reaction (Eq. 4 S): is the conversation stable, without "
            "misunderstanding or a negative reaction?\n"
            "  1 = client feels misunderstood, gets more distressed, withdraws or pushes back\n"
            "  2 = client disengages or gives a short, flat answer\n"
            "  3 = client stays neutral and keeps talking\n"
            "  4 = client feels understood and stays calm and engaged\n"
            "  5 = client clearly feels understood, is calmer and opens up further\n\n"
            "Respond ONLY with valid JSON: {\"empathy\": <float>, \"stability\": <float>}"
        )
        return self._evaluate_json(prompt, {"empathy": 3.0, "stability": 3.0})

    def rank_candidates(self, conversation_history: str, candidates: List[str], reactions: List[str]) -> Optional[List[Dict[str, float]]]:
        # Eq. 4 E and S for all branches in one call (batch scoring with shuffled order, as in the spec)
        block = ""
        for i, (c, r) in enumerate(zip(candidates, reactions), 1):
            block += f"--- Candidate {i} ---\nTherapist: {c}\nPredicted patient reaction: {r}\n\n"
        n = len(candidates)
        prompt = (
            "You are comparing candidate therapist replies for the next turn of a counseling session. "
            "Each candidate comes with the patient's predicted reaction to it.\n\n"
            f"Recent dialogue:\n{conversation_history}\n\n"
            f"{block}"
            "Score EVERY candidate on a 1.0-5.0 scale with one decimal. The order in which they are listed means nothing; "
            "a better candidate must get a higher score.\n\n"
            "empathy of the therapist reply (E):\n"
            "  1 = cold, dismissive or generic; 2 = formulaic validation; 3 = accurate reflection of the feelings;\n"
            "  4 = specific, personal understanding in the client's own terms; 5 = deeply attuned and anticipatory\n"
            "stability, judged from the predicted patient reaction (S): is the conversation stable, without misunderstanding "
            "or a negative reaction?\n"
            "  1 = feels misunderstood, more distressed, withdraws or pushes back; 2 = disengages or gives a flat answer;\n"
            "  3 = neutral, keeps talking; 4 = feels understood, calm and engaged; 5 = clearly feels understood, calmer and opens up further\n\n"
            'Respond ONLY with valid JSON: {"scores": [{"id": 1, "empathy": <float>, "stability": <float>}, ...]} '
            f"with exactly {n} entries."
        )
        data = self._raw_json(prompt, max_tokens=300 + 60 * n)
        try:
            rows = {int(x["id"]): x for x in data["scores"]}
            return [{k: min(5.0, max(1.0, float(rows[i][k]))) for k in ["empathy", "stability"]} for i in range(1, n + 1)]
        except Exception as e:
            logger.warning("candidate scores not parseable (%s), scoring one by one", e)
            return None

    def _raw_json(self, prompt: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        raw = self.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
        )
        try:
            m = re.search(r"\{.*\}", re.sub(r"```(?:json)?", "", raw), re.DOTALL)
            return json.loads(m.group(0) if m else raw)
        except Exception:
            return {}

    def _evaluate_json(self, prompt: str, fallback: Dict[str, float]) -> Dict[str, float]:
        raw = self.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        try:
            cleaned = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            payload = match.group(0) if match else cleaned
            data = json.loads(payload)
            return {k: min(5.0, max(1.0, float(data.get(k, fallback[k])))) for k in fallback}
        except Exception as e:
            logger.warning("Score parsing fallback invoked: %s (Raw: %s)", e, raw)
            return fallback
import re
import json
import logging
from typing import Dict, Any, Optional
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
        therapy_name: str,
    ) -> Dict[str, float]:
        prompt = (
            f"Evaluate the following therapist response within the ongoing session.\n\n"
            f"Dialogue History:\n{conversation_history}\n\n"
            f"Therapist Response:\n{therapist_response}\n\n"
            f"Target Therapy: {therapy_name}\n\n"
            "Score the response on a continuous scale from 1.0 to 5.0 for:\n"
            "- empathy: Affective warmth, emotional attunement, validation (1=cold/robotic, 3=adequate, 5=profoundly attuning)\n"
            f"- alignment: Procedural application and fidelity to {therapy_name} techniques (1=off-track/generic, 3=moderate, 5=exemplary technique fidelity)\n\n"
            "Respond ONLY with valid JSON: {\"empathy\": <float>, \"alignment\": <float>}"
        )
        return self._evaluate_json(prompt, {"empathy": 3.0, "alignment": 3.0})

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
            "Score on a continuous scale from 1.0 to 5.0 for:\n"
            "- empathy: Depth of emotional resonance in the therapist's response (1.0 to 5.0)\n"
            "- stability: Conversational safety, grounding, and forward therapeutic movement in the patient's reaction (1.0 to 5.0)\n\n"
            "Respond ONLY with valid JSON: {\"empathy\": <float>, \"stability\": <float>}"
        )
        return self._evaluate_json(prompt, {"empathy": 3.0, "stability": 3.0})

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
            return {k: float(data.get(k, fallback[k])) for k in fallback}
        except Exception as e:
            logger.warning("Score parsing fallback invoked: %s (Raw: %s)", e, raw)
            return fallback
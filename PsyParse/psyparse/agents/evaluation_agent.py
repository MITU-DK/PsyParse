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
            "Score the response on a continuous scale from 1.0 to 5.0, using one decimal (e.g. 3.6). Use the full range and "
            "do not default to whole numbers; small differences in quality should show up as different scores.\n\n"
            "empathy (Eq. 3 E):\n"
            "  1 = cold, dismissive or generic; ignores what the client said\n"
            "  2 = polite but formulaic validation (\"that sounds hard\")\n"
            "  3 = accurate reflection of the client's feelings\n"
            "  4 = reflects the specific feelings AND their meaning for this client, in the client's own terms\n"
            "  5 = deeply attuned, anticipates what the client may feel next, makes the client feel truly understood\n"
            f"alignment with {therapy_name} (Eq. 3 A):\n"
            "  1 = no recognizable technique, off-track, or invents facts about the client\n"
            "  2 = mentions a technique or gives generic advice without doing it\n"
            "  3 = starts a technique but stays shallow\n"
            "  4 = clearly carries out a technique, tied to the client's actual details\n"
            "  5 = skillful, step-by-step execution that moves the client forward in the plan\n\n"
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
            "Score on a continuous scale from 1.0 to 5.0, using one decimal (e.g. 3.6). Use the full range and "
            "do not default to whole numbers; small differences in quality should show up as different scores.\n\n"
            "empathy of the therapist's response (Eq. 4 E):\n"
            "  1 = cold, dismissive or generic\n"
            "  2 = formulaic validation\n"
            "  3 = accurate reflection of the client's feelings\n"
            "  4 = specific, personal understanding in the client's own terms\n"
            "  5 = deeply attuned and anticipatory\n"
            "stability, judged from the predicted patient reaction (Eq. 4 S):\n"
            "  1 = client withdraws, feels misunderstood, gets more distressed or pushes back\n"
            "  2 = client disengages or gives a short, flat answer\n"
            "  3 = client stays neutral and keeps talking\n"
            "  4 = client opens up, shares more, or engages with the intervention\n"
            "  5 = client gains insight, feels calmer, or commits to a concrete step\n\n"
            "Respond ONLY with valid JSON: {\"empathy\": <float>, \"stability\": <float>}"
        )
        return self._evaluate_json(prompt, {"empathy": 3.0, "stability": 3.0})

    def score_trajectory(self, rollout_dialogue: str, therapy_name: str) -> Dict[str, float]:
        # how the simulated patient actually took the rollout, used to tell therapies apart in stage 3a
        prompt = (
            f"Below is a short simulated counseling exchange using {therapy_name}.\n\n"
            f"{rollout_dialogue}\n\n"
            "Judge ONLY the patient's reactions. Score on a continuous scale from 1.0 to 5.0:\n"
            "- receptivity: does the patient open up, engage with the intervention and move forward "
            "(1=resists, feels misunderstood or withdraws, 3=neutral, 5=clearly engaged and gaining insight)\n\n"
            "Respond ONLY with valid JSON: {\"receptivity\": <float>}"
        )
        return self._evaluate_json(prompt, {"receptivity": 3.0})

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
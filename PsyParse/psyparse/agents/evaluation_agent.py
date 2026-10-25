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

    def rank_candidates(self, conversation_history: str, candidates: List[str], reactions: List[str]) -> Optional[List[Dict[str, float]]]:
        # Eq. 4 scores for all branches in ONE call, plus a forced best-to-worst ranking. scores alone tied in ~half
        # the turns (all 4.0) and otherwise just went down with the list position (4.5, 4.4, 4.3, 4.2)
        block = ""
        for i, (c, r) in enumerate(zip(candidates, reactions), 1):
            block += f"--- Candidate {i} ---\nTherapist: {c}\nPredicted patient reaction: {r}\n\n"
        n = len(candidates)
        prompt = (
            "You are comparing candidate therapist replies for the next turn of a counseling session. "
            "Each candidate comes with the patient's predicted reaction to it.\n\n"
            f"Recent dialogue:\n{conversation_history}\n\n"
            f"{block}"
            "First read all candidates and decide which one is best for the client right now, considering what each one "
            "actually does (which technique, how specific to this client, what it adds compared to the earlier turns). "
            "The order in which they are listed means nothing. Put them in a \"ranking\" from best to worst: every id "
            "exactly once, no ties. Then score every candidate on a 1.0-5.0 scale with one decimal, consistent with your ranking.\n\n"
            "empathy of the therapist reply (E):\n"
            "  1 = cold, dismissive or generic; 2 = formulaic validation; 3 = accurate reflection of the feelings;\n"
            "  4 = specific, personal understanding in the client's own terms; 5 = deeply attuned and anticipatory\n"
            "stability, judged from the predicted patient reaction (S):\n"
            "  1 = withdraws, feels misunderstood or more distressed; 2 = disengages or gives a flat answer; 3 = neutral, keeps talking;\n"
            "  4 = opens up or engages with the intervention; 5 = gains insight, feels calmer or commits to a concrete step\n"
            "therapeutic value of the therapist reply (G):\n"
            "  1 = no real help: empty praise, stock phrases, repeats earlier turns or keeps polishing a task that is already done;\n"
            "  2 = mostly questions or reflection; 3 = some useful content;\n"
            "  4 = substantive help tied to this client's details (insight, a technique carried out, a concrete step);\n"
            "  5 = rich, personal help that covers feelings, thoughts and actions and clearly moves the client toward their goals\n\n"
            'Respond ONLY with valid JSON: {"ranking": [<best id>, ..., <worst id>], '
            '"scores": [{"id": 1, "empathy": <float>, "stability": <float>, "goal": <float>}, ...]} '
            f"with exactly {n} ids in the ranking and {n} score entries."
        )
        data = self._raw_json(prompt, max_tokens=300 + 80 * n)
        try:
            rows = {int(x["id"]): x for x in data["scores"]}
            res = []
            for i in range(1, n + 1):
                e, st = float(rows[i]["empathy"]), float(rows[i]["stability"])
                g = float(rows[i].get("goal", (e + st) / 2))
                res.append({k: min(5.0, max(1.0, v)) for k, v in [("empathy", e), ("stability", st), ("goal", g)]})
        except Exception as e:
            logger.warning("candidate ranking not parseable (%s), scoring one by one", e)
            return None
        try:
            order = [int(x) for x in data.get("ranking", [])]
        except Exception:
            order = []
        if sorted(order) == list(range(1, n + 1)):
            for pos, i in enumerate(order, 1):
                res[i - 1]["rank"] = float(pos)
        else:
            # no usable ranking, derive it from the scores (equal scores share a rank)
            tot = [sum(r.values()) for r in res]
            for i, r in enumerate(res):
                r["rank"] = float(1 + sum(1 for t in tot if t > tot[i]))
        return res

    def compare_trajectories(self, rollouts: Dict[str, str]) -> Optional[Dict[str, Dict[str, float]]]:
        # stage 3a: put the simulated sessions next to each other and score them relative to each other
        names = list(rollouts)
        block = ""
        for i, th in enumerate(names, 1):
            block += f"=== Session {i} ({th}) ===\n{rollouts[th]}\n\n"
        prompt = (
            "Below are short simulated counseling sessions with the SAME client, each using a different therapy.\n\n"
            f"{block}"
            "Compare the sessions against each other and score each one on a continuous 1.0-5.0 scale with one decimal. "
            "The better session must get the higher score; give equal scores only if they are truly equally good.\n"
            "- empathy: how understood the client is (1 = generic, 5 = deeply attuned to this client)\n"
            "- alignment: how clearly and correctly the therapist carries out techniques of its own therapy "
            "(1 = no recognizable technique, 3 = shallow or only mentioned, 5 = skillful step-by-step execution)\n"
            "- progress: how far the client moves from describing the problem toward new insight, a reframe or a concrete, "
            "committed step (1 = stays in a listening/validation loop, 5 = clear insight and a concrete commitment)\n\n"
            'Respond ONLY with valid JSON: {"sessions": [{"id": 1, "empathy": <float>, "alignment": <float>, "progress": <float>}, ...]} '
            f"with exactly {len(names)} entries."
        )
        data = self._raw_json(prompt, max_tokens=300 + 80 * len(names))
        try:
            rows = {int(x["id"]): x for x in data["sessions"]}
            return {th: {k: min(5.0, max(1.0, float(rows[i][k]))) for k in ["empathy", "alignment", "progress"]}
                    for i, th in enumerate(names, 1)}
        except Exception as e:
            logger.warning("trajectory comparison not parseable: %s", e)
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
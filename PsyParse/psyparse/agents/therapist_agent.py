import re
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
import config
from psyparse.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# short named techniques per therapy, the db "techniques" are long paragraphs so these give the llm clean anchors
KNOWN_TECHNIQUES = {
    "CBT": ["cognitive restructuring", "thought record", "behavioral activation", "Socratic questioning"],
    "SFBT": ["miracle question", "scaling questions", "exception finding", "small next step"],
    "PCT": ["reflective listening", "empathic understanding", "unconditional positive regard"],
    "ACT": ["cognitive defusion", "values clarification", "acceptance of difficult feelings", "committed action"],
    "REBT": ["ABC model", "disputing irrational beliefs", "rational coping statements"],
    "Psychoanalytic": ["exploring past relational patterns", "interpretation of recurring themes", "free association"],
    "Adlerian": ["lifestyle assessment", "encouragement", "early recollections"],
    "Family": ["mapping family interactions", "reframing family roles", "communication rehearsal"],
    "Family Systems": ["genogram exploration", "differentiation of self", "triangulation awareness"],
    "Gestalt": ["here-and-now awareness", "empty chair dialogue", "body awareness"],
    "Reality": ["WDEP questions (wants, doing, evaluation, plan)", "choice theory reframing"],
    "Postmodern": ["externalizing the problem", "re-authoring the story", "unique outcomes"],
    "Existential": ["meaning exploration", "responsibility and choice", "facing uncertainty"],
}

def fallback_slice(cand: Dict[str, Any]) -> Dict[str, Any]:
    # only used when the llm gives no usable framework for a therapy
    th = cand.get("therapy_type", "CBT")
    techs = KNOWN_TECHNIQUES.get(th) or [t[:120] for t in cand.get("techniques", [])[:2]] or ["supportive exploration"]
    return {
        "therapy": th,
        "techniques": techs[:3],
        "procedural_steps": [f"Apply {t} to the client's problems" for t in techs[:3]],
    }


class TherapistAgent(BaseAgent):
    """
    Therapist Agent orchestrating:
      1) Stage 1: Structured Diagnostic Interview via Question Matrix.
      2) Stage 2: Synthesis of a Multi-Therapy Guidance Framework.
      3) Stage 3a: Technique-Guided Multi-Turn Rollout.
      4) Stage 3b: Response Branch Generation.
    """
    def __init__(
        self,
        mode: str = "interview",
        model: Optional[str] = None,
        temperature: float = 0.7,
    ):
        system_prompt = self._get_interview_prompt() if mode == "interview" else "You are an expert psychological counselor."
        super().__init__(system_prompt=system_prompt, model=model, temperature=temperature)

    def _get_interview_prompt(self) -> str:
        # paper stage 1: "a set of baseline counseling strategies" = the same plain counselor prompt as the baseline
        return config.COUNSELOR_PROMPT

    def conduct_interview_turn(self, target_question: str, patient_last_response: Optional[str] = None) -> str:
        # paper: intake uses "baseline counseling strategies", so reply like a normal counselor and weave the question in
        # (only asking the question gave ~60 word replies vs ~350 for the baseline)
        instruction = ("Respond to the client as you normally would as their counselor, then incorporate this target "
                       f"question naturally in the same reply: {target_question}")
        msg = f"{patient_last_response}\n\n[Instruction: {instruction}]" if patient_last_response else f"[Instruction: {instruction}]"
        return self.send(msg)

    def synthesize_guidance(self, top_therapies: List[Dict[str, Any]], patient_profile: Dict[str, Any]) -> Dict[str, Any]:
        names = [t.get("therapy_type", "CBT") for t in top_therapies]
        cands = []
        for t in top_therapies:
            cands.append({
                "therapy": t.get("therapy_type", "CBT"),
                "techniques_from_similar_case": [x[:120] for x in t.get("techniques", [])[:3]],
                "applicable_conditions": t.get("applicable_conditions", [])[:3],
            })

        def ask(cs, ns):
            prompt = (
                "You are a master clinical supervisor. Build a personalized counseling guidance framework "
                "for the following patient profile:\n"
                f"{json.dumps(patient_profile, indent=2)}\n\n"
                f"Retrieved candidate therapies:\n{json.dumps(cs, indent=2)}\n\n"
                "For each therapy give the therapy type, its core techniques for this patient and the procedural steps. "
                "You must rely on your internal clinical knowledge to actively generate step-by-step procedural_steps for how "
                "to apply these techniques to this patient.\n"
                f"Write exactly ONE framework for EACH of these therapies: {json.dumps(ns)}. "
                "The \"therapy\" field must be exactly the given name (do not merge or rename therapies).\n"
                "Output strictly valid JSON with this exact schema:\n"
                "{\n"
                '  "frameworks": [\n'
                '    {"therapy": "<name>", "techniques": ["..."], "procedural_steps": ["Step 1: ...", "Step 2: ..."]}\n'
                "  ]\n"
                "}\n"
                "Respond ONLY with valid JSON."
            )
            raw = self.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=1000 + 800 * len(ns),
            )
            try:
                data = json.loads(self._clean_json(raw))
                return [f for f in data.get("frameworks", []) if isinstance(f, dict)]
            except Exception as e:
                logger.error("Failed to parse guidance JSON: %s", e)
                return []

        def find(got, name):
            # exact name first, "family" is also inside "family systems"
            others = {n.lower() for n in names if n.lower() != name.lower()}
            exact = [f for f in got if str(f.get("therapy", "")).strip().lower() == name.lower()]
            # loose match ("CBT" in "CBT framework") but never steal another requested therapy's plan
            loose = [f for f in got if name.lower() in str(f.get("therapy", "")).lower() and f not in exact
                     and str(f.get("therapy", "")).strip().lower() not in others]
            for f in exact + loose:
                f = dict(f)
                # small models sometimes give one string instead of a list
                for k in ["techniques", "procedural_steps"]:
                    if isinstance(f.get(k), str):
                        f[k] = [x.strip() for x in re.split(r"\n|;|(?=Step \d)", f[k]) if x.strip()]
                    f[k] = [str(x) for x in (f.get(k) or []) if str(x).strip()]
                if f["techniques"] and len(f["procedural_steps"]) >= 3:
                    return f
            return None

        got = ask(cands, names)
        res = []
        for cand, c, name in zip(top_therapies, cands, names):
            match = find(got, name)
            if match is None:
                # small models often skip the 2nd/3rd therapy, ask again for just this one
                match = find(ask([c], [name]), name)
            if match is not None:
                match["therapy"] = name
                res.append(match)
            else:
                logger.warning("no usable framework for %s, using fallback", name)
                res.append(fallback_slice(cand))
        return {"frameworks": res}

    def setup_rollout_mode(self, therapy_slice: Dict[str, Any], patient_profile: Dict[str, Any]) -> None:
        # spec guided-response prompt: only this therapy's slice of the guidance framework
        th_name = therapy_slice.get("therapy", "CBT")
        techs = ", ".join(therapy_slice.get("techniques", []))
        steps = "\n".join(therapy_slice.get("procedural_steps", []))
        guided_prompt = (
            f"You are a psychological counselor. Using {th_name} techniques, specifically {techs}, following these "
            f"procedural steps:\n{steps}\n"
            f"and keeping in mind the patient's profile: {json.dumps(patient_profile)}, respond to the patient.\n"
            "You already had an intake conversation with this patient (it is in the conversation history); continue the "
            "session from there and follow the steps in order over the session. Only use facts the patient actually said."
        )
        self.swap_system_prompt(guided_prompt)

    def respond_step(self, turn: int, max_turns: int, temperature: float = 0.7) -> str:
        instruction = (
            f"\n\nSession progress: therapy turn {turn} of about {max_turns}.\n"
            "Output ONLY the counselor's spoken words."
        )
        resp = self.generate(
            messages=self.get_history(),
            temperature=temperature,
            ephemeral_system=self.system_prompt + instruction,
        )
        resp = re.sub(r"^\s*\**(counselor|therapist)\**\s*:\s*", "", resp, flags=re.IGNORECASE)
        return resp.strip().strip('"').strip()

    def generate_diverse_branches(self, turn: int, max_turns: int, n: int = 4, temperature: float = 0.7) -> List[str]:
        # paper: n candidates from the same prompt, diversity only from T_single
        # respond_step doesnt touch history so the branches can be generated at the same time
        with ThreadPoolExecutor(max_workers=max(1, min(config.PARALLEL_CALLS, n))) as ex:
            return list(ex.map(lambda i: self.respond_step(turn, max_turns, temperature=temperature), range(n)))

    @staticmethod
    def _clean_json(text: str) -> str:
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else text

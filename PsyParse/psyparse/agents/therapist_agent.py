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

ANGLES = [
    "Cognitive angle: gently examine or reframe one specific thought the client just expressed.",
    "Emotional angle: deeply validate the specific feeling underneath what the client just said.",
    "Behavioral angle: propose one concrete, small action, exercise or experiment fitted to the client's situation.",
    "Socratic angle: ask one reflective question that helps the client discover something new about their situation.",
]


def fallback_slice(cand: Dict[str, Any]) -> Dict[str, Any]:
    th = cand.get("therapy_type", "CBT")
    techs = KNOWN_TECHNIQUES.get(th) or [t[:120] for t in cand.get("techniques", [])[:2]] or ["supportive exploration"]
    return {
        "therapy": th,
        "techniques": techs[:3],
        "procedural_steps": [
            "Validate the client's distress and summarize what they shared",
            "Explore the key thoughts and patterns behind it",
            f"Apply {techs[0]} to the client's main problem",
            "Agree on one concrete next step and consolidate insight",
        ],
    }


def current_step(steps: List[str], turn: int, max_turns: int) -> str:
    if not steps:
        return ""
    idx = min(len(steps) - 1, (turn - 1) * len(steps) // max(1, max_turns))
    return steps[idx]


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
        return (
            "You are a professional psychological counselor conducting an initial structured intake assessment. "
            "Your objective is to explore the patient's core problems, emotional states, symptoms, cognitive distortions, "
            "interpersonal issues, and therapeutic goals with warmth, validation, and active listening. "
            "Ask exactly ONE clear, empathetic, and open-ended target question per turn. "
            "Output only what you say to the client, in 3-5 natural spoken sentences (no lists or headings)."
        )

    def conduct_interview_turn(self, target_question: str, patient_last_response: Optional[str] = None) -> str:
        instruction = (
            f"Target assessment focus for this turn: '{target_question}'.\n"
            "First reflect back what the patient just shared, using their own words. Then add one short validating or "
            "normalizing sentence (e.g. why this reaction makes sense given what they described). Then naturally ask the target question. "
            "Do NOT ask multiple questions and do not give advice yet. Only use facts the patient actually said."
        )
        if patient_last_response:
            return self.send(f"{patient_last_response}\n\n[Instruction: {instruction}]")
        return self.send(f"[Instruction: {instruction}]")

    def synthesize_guidance(self, top_therapies: List[Dict[str, Any]], patient_profile: Dict[str, Any]) -> Dict[str, Any]:
        names = [t.get("therapy_type", "CBT") for t in top_therapies]
        cands = []
        for t in top_therapies:
            cands.append({
                "therapy": t.get("therapy_type", "CBT"),
                "core_techniques": KNOWN_TECHNIQUES.get(t.get("therapy_type"), []),
                "techniques_from_similar_case": [x[:120] for x in t.get("techniques", [])[:3]],
                "applicable_conditions": t.get("applicable_conditions", [])[:3],
            })

        prompt = (
            "You are a master clinical supervisor. Build a personalized counseling guidance framework "
            "for the following patient profile:\n"
            f"{json.dumps(patient_profile, indent=2)}\n\n"
            f"Retrieved candidate therapies:\n{json.dumps(cands, indent=2)}\n\n"
            "Requirements:\n"
            f"1. Write exactly ONE framework for EACH of these therapies: {json.dumps(names)}. "
            "The \"therapy\" field must be exactly the given name (do not merge or rename therapies).\n"
            "2. \"techniques\": 2-4 short, named techniques of that therapy that suit THIS patient.\n"
            "3. \"procedural_steps\": 4-5 ordered steps from opening to consolidation. Each step must say what to do "
            "and mention the patient's specific problem, thought or goal it targets.\n"
            "Output strictly valid JSON with this exact schema:\n"
            "{\n"
            '  "frameworks": [\n'
            '    {"therapy": "<name>", "techniques": ["..."], "procedural_steps": ["Step 1: ...", "Step 2: ..."]}\n'
            "  ]\n"
            "}\n"
            "Respond ONLY with valid JSON."
        )
        data = {}
        raw = self.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=1500,
        )
        try:
            data = json.loads(self._clean_json(raw))
        except Exception as e:
            logger.error("Failed to parse guidance JSON: %s. Using fallback for all therapies.", e)

        got = [f for f in data.get("frameworks", []) if isinstance(f, dict)] if isinstance(data, dict) else []
        res = []
        for cand, name in zip(top_therapies, names):
            match = None
            for f in got:
                fname = str(f.get("therapy", "")).lower()
                if fname == name.lower() or name.lower() in fname:
                    match = f
                    break
            if match and match.get("techniques") and match.get("procedural_steps"):
                match["therapy"] = name
                res.append(match)
            else:
                logger.warning("no usable framework for %s, using fallback", name)
                res.append(fallback_slice(cand))
        return {"frameworks": res}

    def setup_rollout_mode(self, therapy_slice: Dict[str, Any], patient_profile: Dict[str, Any]) -> None:
        th_name = therapy_slice.get("therapy", "CBT")
        techs = ", ".join(therapy_slice.get("techniques", []))
        steps = " -> ".join(therapy_slice.get("procedural_steps", []))
        guided_prompt = (
            f"You are a professional counselor conducting an active therapy session using {th_name}.\n"
            f"Target Techniques: {techs}\n"
            f"Procedural Plan: {steps}\n"
            f"Patient Context: {json.dumps(patient_profile)}\n\n"
            "You already did an intake interview with this client (it is in the conversation history).\n\n"
            "How to respond:\n"
            "- Do real therapeutic work in every reply, not just reflection. Briefly reflect what the client just said, "
            "then actively carry out the current technique WITH the client: e.g. walk through a thought record step by step "
            "(situation, automatic thought, evidence for and against, balanced thought), ask the scaling question and follow up on "
            "the number, guide a short grounding or behavioral exercise, or explore a value and link it to an action.\n"
            "- Explain briefly why the technique fits what the client described, so it feels personal, not generic.\n"
            "- Usually 3-6 sentences. Speak naturally as in a real session: no headings, bullet lists, tables or markdown.\n"
            "- End with at most one question or one concrete invitation to try something.\n"
            "- Only use facts the client actually said in this conversation. Never invent past events, people, names or "
            "details. If a technique needs examples from the client's life (e.g. exceptions or past successes), ASK for them."
        )
        self.swap_system_prompt(guided_prompt)

    def respond_step(
        self,
        therapy_slice: Dict[str, Any],
        turn: int,
        max_turns: int,
        temperature: float = 0.7,
        angle: Optional[str] = None,
    ) -> str:
        steps = therapy_slice.get("procedural_steps", [])
        step = current_step(steps, turn, max_turns)
        th_name = therapy_slice.get("therapy", "CBT")
        instruction = (
            f"\n\nSession progress: therapy turn {turn} of about {max_turns}.\n"
            f"Current step of the {th_name} plan: {step}\n"
            "Respond to what the client just said, connect it to a specific detail they shared earlier, "
            "and actually carry out the current step with one of the target techniques (do it, don't just name it)."
        )
        if turn >= max_turns:
            instruction += (
                "\nThis is the END of the session: summarize in the client's own words what they noticed or learned today, "
                "acknowledge their effort, and agree together on one small, concrete next step to try before the next session."
            )
        elif max_turns >= 6 and turn == max_turns - 1:
            instruction += "\nThe session is nearing its end: start consolidating what the client has discovered so far."
        if angle:
            instruction += f"\nFrame your reply with this angle: {angle}"
        instruction += "\nOutput ONLY the counselor's spoken words."

        resp = self.generate(
            messages=self.get_history(),
            temperature=temperature,
            ephemeral_system=self.system_prompt + instruction,
        )
        resp = re.sub(r"^\s*\**(counselor|therapist)\**\s*:\s*", "", resp, flags=re.IGNORECASE)
        return resp.strip().strip('"').strip()

    def generate_diverse_branches(
        self,
        therapy_slice: Dict[str, Any],
        turn: int,
        max_turns: int,
        n: int = 4,
        temperature: float = 0.7,
    ) -> List[str]:
        # one branch per clinical angle so the 4 candidates are actually different
        # respond_step doesnt touch history so the branches can be generated at the same time
        def one(i):
            return self.respond_step(therapy_slice, turn, max_turns, temperature=temperature, angle=ANGLES[i % len(ANGLES)])
        with ThreadPoolExecutor(max_workers=max(1, min(config.PARALLEL_CALLS, n))) as ex:
            return list(ex.map(one, range(n)))

    @staticmethod
    def _clean_json(text: str) -> str:
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else text

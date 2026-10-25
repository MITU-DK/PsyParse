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

# every angle still carries out the plan step, they only change how the reply gets there
ANGLES = [
    "Lead with the thinking side: carry out the step by working with one specific thought or belief the client just expressed.",
    "Lead with the feeling: first validate the specific emotion underneath what the client said, then carry out the step.",
    "Lead with action: carry out the step through a concrete exercise, experiment or small plan fitted to the client's situation.",
    "Lead with discovery: carry out the step through guided questions that help the client find the insight themselves.",
]


def fallback_slice(cand: Dict[str, Any]) -> Dict[str, Any]:
    th = cand.get("therapy_type", "CBT")
    techs = KNOWN_TECHNIQUES.get(th) or [t[:120] for t in cand.get("techniques", [])[:2]] or ["supportive exploration"]
    return {
        "therapy": th,
        "techniques": techs[:3],
        "procedural_steps": [
            "Link back to what the client shared in the intake and agree on what to focus on today",
            "Explore the key thoughts, feelings and patterns behind the main problem with a concrete recent example",
            f"Explain and carry out {techs[0]} on the client's main problem",
            f"Practice {techs[0]} together on another concrete situation from the client's life",
            f"Use {techs[1] if len(techs) > 1 else techs[0]} on the client's second problem area",
            "Plan one small, concrete step for the coming week (what, when, how) and anticipate obstacles",
            "Summarize what the client discovered today in their own words and consolidate the plan",
        ],
    }


def plan_span(max_turns: int) -> int:
    # the patient may end the session from MIN_THERAPY_TURNS on (8 of 10), so the plan has to be done by then,
    # otherwise the "plan a step" / "consolidate" steps never happen
    return max(1, min(max_turns, config.MIN_THERAPY_TURNS)) if max_turns >= 6 else max(1, max_turns)


def current_step(steps: List[str], turn: int, max_turns: int) -> str:
    if not steps:
        return ""
    idx = min(len(steps) - 1, (turn - 1) * len(steps) // plan_span(max_turns))
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
            "You are an experienced psychological counselor in the first part of a session, getting to know the client "
            "before choosing an approach. You want to understand their core problems, feelings, symptoms, thinking patterns, "
            "relationships and goals. Be warm, curious and natural: this is a real conversation, not a questionnaire. "
            "Each turn, respond to what the client actually just said, give them something helpful back, then explore one topic "
            "with one open question. Output only what you say to the client, in natural spoken sentences (no lists or headings). "
            "Vary your wording from turn to turn."
        )

    def conduct_interview_turn(self, target_question: str, patient_last_response: Optional[str] = None) -> str:
        instruction = (
            f"Topic to explore this turn: {target_question}\n"
            "Start from the actual content of what the client just said: if they mention an event, a person or a feeling, "
            "pick it up specifically (don't just paraphrase it back). Then give one short piece of real help that fits what they "
            "described: a normalizing explanation, a gentle reframe, or a bit of psychoeducation (e.g. how avoidance keeps anxiety going). "
            "Then ask ONE open question that moves toward the topic, in your own words and linked to what they said. "
            "If they already covered this topic, go deeper on it instead of asking it again. "
            "No advice lists yet. Only use facts the client actually said. "
            "Avoid stock openers like 'It sounds like...' or 'It's understandable...'."
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

        def ask(cs, ns):
            prompt = (
                "You are a master clinical supervisor. Build a personalized counseling guidance framework "
                "for the following patient profile:\n"
                f"{json.dumps(patient_profile, indent=2)}\n\n"
                f"Retrieved candidate therapies:\n{json.dumps(cs, indent=2)}\n\n"
                "Requirements:\n"
                f"1. Write exactly ONE framework for EACH of these therapies: {json.dumps(ns)}. "
                "The \"therapy\" field must be exactly the given name (do not merge or rename therapies).\n"
                "2. \"techniques\": 2-4 short, named techniques of that therapy that suit THIS patient.\n"
                f"3. \"procedural_steps\": 6-7 ordered steps that fill a therapy session of about {config.THERAPY_TURNS} turns: "
                "link to the intake and set the focus, explore the key pattern with a concrete example, carry out the main technique, "
                "practice it on another real situation, work on the patient's second problem area or a second technique, "
                "plan a concrete step for the week, consolidate. Each step must say what to do and name the patient's specific "
                "problem, thought or goal it targets.\n"
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
                # small models sometimes give one string instead of a list, then current_step would pick single characters
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
        th_name = therapy_slice.get("therapy", "CBT")
        techs = ", ".join(therapy_slice.get("techniques", []))
        steps = " -> ".join(therapy_slice.get("procedural_steps", []))
        guided_prompt = (
            f"You are an experienced counselor running an active therapy session using {th_name}.\n"
            f"Techniques to use: {techs}\n"
            f"Session plan: {steps}\n"
            f"What you know about the client: {json.dumps(patient_profile)}\n\n"
            "You already had an intake conversation with this client (it is in the conversation history).\n\n"
            "How to respond:\n"
            "- Every reply does real therapeutic work. First show you understood what the client just said, in their own terms. "
            "Then carry out the current step of the plan WITH the client: name the technique in plain words, say briefly why it "
            "fits what they described, and actually do it, e.g. walk through a thought record step by step (situation, automatic "
            "thought, evidence for and against, balanced thought), ask the scaling question and follow up on the number, guide a "
            "short grounding or breathing exercise, explore a value and link it to an action, or plan a small experiment "
            "(what, when, where, what might get in the way).\n"
            "- Tie what you say to specific details the client shared earlier (people, events, their own words), so it is clearly "
            "about them and not generic.\n"
            "- Take the space the moment needs: a short paragraph when exploring, more when explaining or walking through an "
            "exercise or plan (usually about 80-200 words). When you give an exercise or a plan you may lay out the steps as a "
            "short numbered list; otherwise talk naturally, without headings.\n"
            "- End with one clear question or one concrete invitation to try something. Avoid stock phrases like "
            "'How does that sound?' or 'every step of the way', and don't repeat what you already said in earlier turns.\n"
            "- Only use facts the client actually said. Never invent events, people, names or details; if you need an example "
            "from their life, ask for it.\n"
            "- If the client starts saying goodbye before the plan is finished, acknowledge it warmly and keep working on the "
            "current step."
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
            "and actually carry out the current step with one of the target techniques (do it, don't just name it). "
            "Move the session forward: don't redo a step you already finished in earlier turns."
        )
        if turn >= max_turns:
            instruction += (
                "\nThis is the END of the session: summarize in the client's own words what they noticed or learned today, "
                "acknowledge their effort, and agree together on one small, concrete next step to try before the next session."
            )
        elif turn > plan_span(max_turns):
            # plan is done but the session goes on: deepen instead of looping on goodbyes
            instruction += (
                "\nThe plan is basically done. Use this turn to deepen the work instead of summarizing again: rehearse the "
                "planned step together (e.g. a short role-play or walking through the exact moment), work through an obstacle "
                "that could get in the way, or pick up something the client raised that is still open."
            )
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

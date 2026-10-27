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

# stock openers the 14b model keeps using even when the prompt bans them (59% of therapy turns in run 5)
STOCK = re.compile(r"^\W*(it sounds like|that sounds like|that's (a |so |really )?(great|fantastic|wonderful|brilliant)|"
                   r"it's (so |really |incredibly )?(wonderful|great|meaningful)|i'm (so |really |very )?(glad|thrilled)|it's (completely |totally |perfectly )?(normal|understandable)|you're (very |so )?welcome|"
                   r"it's (really |incredibly |so )?(tough|brave|hard)|i'm (so |really )?sorry to hear|i (can |completely |totally )?understand)", re.I)


def is_stock(txt: str) -> bool:
    return bool(STOCK.match((txt or "").replace("\u2019", "'").strip()))


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


def support_for(guidance: Dict[str, Any], th: str) -> List[str]:
    # techniques of the other retrieved therapies back up the chosen plan (multi-therapy guidance)
    res = []
    for f in guidance.get("frameworks", []):
        if str(f.get("therapy", "")).lower() != str(th).lower():
            res += [f"{t} ({f.get('therapy')})" for t in f.get("techniques", [])[:2]]
    return res


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
            "You are an experienced psychological counselor in the first part of a session. While you get to know the client "
            "(core problems, feelings, symptoms, thinking patterns, relationships, goals) you already counsel them: every reply "
            "should leave the client with something of value, not just a question. Be warm, specific and natural. "
            "Output only what you say to the client. Vary your wording from turn to turn."
        )

    def conduct_interview_turn(self, target_question: str, patient_last_response: Optional[str] = None) -> str:
        instruction = (
            f"Topic to explore this turn: {target_question}\n"
            "Write a full counseling reply: (1) respond to the specific content of what the client just said (events, people, "
            "feelings, their exact words) and validate the feeling underneath; (2) give real help that fits it: explain what may "
            "be going on (e.g. how a thought pattern or avoidance keeps the problem alive) and offer one concrete thing they can "
            "notice or try; (3) then ask ONE open question that explores the topic above, in your own words and linked to what they "
            "said. If they already covered this topic, go deeper on it instead of asking it again. "
            "Only use facts the client actually said. Don't open with a stock phrase or praise ('It sounds like...', "
            "'It's understandable...', 'It's really tough...', 'That's a great...'); start with the substance."
        )
        msg = f"{patient_last_response}\n\n[Instruction: {instruction}]" if patient_last_response else f"[Instruction: {instruction}]"
        res = self.send(msg)
        if is_stock(res):
            # the model ignores the ban on stock openers half the time, one more try
            res = self.generate(self.history[:-1], temperature=0.9)
            self.history[-1]["content"] = res
        return res

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

    def setup_rollout_mode(self, therapy_slice: Dict[str, Any], patient_profile: Dict[str, Any],
                           support: Optional[List[str]] = None) -> None:
        th_name = therapy_slice.get("therapy", "CBT")
        self.support = support or []
        techs = ", ".join(therapy_slice.get("techniques", []))
        # multi-therapy guidance (paper): techniques of the other retrieved therapies can back up the main plan
        extra = f"Supporting techniques from the other retrieved therapies (use when they fit): {', '.join(support)}\n" if support else ""
        steps = " -> ".join(therapy_slice.get("procedural_steps", []))
        guided_prompt = (
            f"You are an experienced counselor running an active therapy session using {th_name}.\n"
            f"Techniques to use: {techs}\n"
            f"Session plan: {steps}\n"
            f"{extra}"
            f"What you know about the client: {json.dumps(patient_profile)}\n\n"
            "You already had an intake conversation with this client (it is in the conversation history).\n\n"
            "How to respond:\n"
            "- Every reply is a full, substantive counseling reply that covers the whole person: (1) the feeling: validate the "
            "specific emotion in the client's own terms; (2) the thinking: work on the meaning, thought or pattern behind it with "
            "the current step of the plan; (3) the doing: when it fits, give a concrete tool, exercise or small step, tied to the "
            "client's real situation.\n"
            "- Actually carry out the current step WITH the client instead of describing it, e.g. walk through a thought record "
            "step by step (situation, automatic thought, evidence for and against, balanced thought), ask the scaling question and "
            "follow up on the number, guide a short grounding or breathing exercise, explore a value and link it to an action, or "
            "plan a small experiment (what, when, where, what might get in the way). Briefly explain why it fits what they described.\n"
            "- Never announce what you are doing ('I'll use empathic understanding to...', 'Let me apply reflective listening'); "
            "just do it. Naming a technique is fine when you teach it to the client.\n"
            "- Tie what you say to specific details the client shared earlier (people, places, events, their own words), so it is "
            "clearly about them and not generic.\n"
            "- Be as thorough as a skilled counselor in a written session: explain, teach and guide, don't just ask. When you give "
            "an exercise or a plan, lay out the steps clearly (a short numbered list is fine).\n"
            "- End with one clear question or one concrete invitation to try something. Avoid stock phrases like "
            "'How does that sound?' or 'every step of the way', and don't repeat what you already said in earlier turns.\n"
            "- Don't open with praise or a stock phrase ('That's a great...', 'It's wonderful to hear', 'I'm glad to hear', "
            "'It sounds like', 'It's completely normal', 'It's incredibly brave'); start with the substance.\n"
            "- Once a task is settled (a message is written, a step is planned), don't keep polishing it in later turns; "
            "move on to the next step of the plan.\n"
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
        focus: Optional[str] = None,
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
        if focus:
            instruction += f"\nFor this reply, build the work on the current step around this technique: {focus}."
        if turn >= max_turns:
            instruction += (
                "\nThis is the END of the session: summarize in the client's own words what they noticed or learned today, "
                "acknowledge their effort, and agree together on one small, concrete next step to try before the next session."
            )
        elif turn > plan_span(max_turns):
            # plan is done but the session goes on: deepen instead of looping on goodbyes
            instruction += (
                "\nThe plan's steps are done. Don't polish the same task again and don't summarize again. Pick a problem from "
                "the intake that has not been worked on yet (see what you know about the client: core problems, thinking "
                "patterns, relationships) and work on it with one of the techniques. If everything was covered, add something "
                "new to the main problem: rehearse the hard moment (short role-play), plan for a setback, or link it to a value."
            )
        instruction += "\nOutput ONLY the counselor's spoken words."

        resp = ""
        for t in range(2):
            resp = self.generate(
                messages=self.get_history(),
                temperature=temperature if t == 0 else min(1.0, temperature + 0.2),
                ephemeral_system=self.system_prompt + instruction,
            )
            resp = re.sub(r"^\s*\**(counselor|therapist)\**\s*:\s*", "", resp, flags=re.IGNORECASE).strip().strip('"').strip()
            if not is_stock(resp):
                break
            # half the drafts open with "I'm glad..." / "That's a great..." despite the ban, name it and try once more
            opener = " ".join(resp.split()[:4])
            instruction += f"\nDo NOT start with '{opener}' or any other praise or stock phrase. Start directly with the substance."
        return resp

    def generate_diverse_branches(
        self,
        therapy_slice: Dict[str, Any],
        turn: int,
        max_turns: int,
        n: int = 4,
        temperature: float = 0.7,
    ) -> List[str]:
        # temperature alone (paper) gave 4 near copies that the scorer couldn't tell apart, and the old "one clinical
        # angle per branch" made replies one-dimensional. so: branch 1 follows the plan as written, the others build the
        # same full reply around a different technique (own therapy + supporting ones), rotated over the turns
        pool = list(therapy_slice.get("techniques", [])) + list(getattr(self, "support", []))
        # respond_step doesnt touch history so the branches can be generated at the same time
        self.last_focus = [pool[(turn * (n - 1) + i - 1) % len(pool)] if i > 0 and pool else None for i in range(n)]

        def one(i):
            return self.respond_step(therapy_slice, turn, max_turns, temperature=temperature, focus=self.last_focus[i])
        with ThreadPoolExecutor(max_workers=max(1, min(config.PARALLEL_CALLS, n))) as ex:
            return list(ex.map(one, range(n)))

    @staticmethod
    def _clean_json(text: str) -> str:
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else text

import re
from typing import Dict, Any, Optional
from psyparse.agents.base_agent import BaseAgent


_openings: Dict[str, str] = {}
_personas: Dict[str, str] = {}


def is_client(t: Dict[str, Any]) -> bool:
    who = f"{t.get('participant', '')} {t.get('speaker', '')}".lower()
    return "client" in who or "seeker" in who


def get_persona(scenario: Dict[str, Any], model: Optional[str] = None) -> str:
    # the dataset background is one short sentence (7-17 words), so the patient used to invent its life story on the fly,
    # differently in each arm. like the paper, build a fuller patient profile from the scenario once: background + the
    # client's own lines from the real session (never the therapist lines or therapy notes). cached, both arms get the same one
    sid = scenario.get("dialog_id", "")
    if sid in _personas:
        return _personas[sid]
    bg = str(scenario.get("background", "")).strip() or "Experiencing emotional and psychological distress."
    said = "\n".join(f"- {str(t.get('content', '')).strip()}" for t in scenario.get("dialog", [])
                     if is_client(t) and str(t.get("content", "")).strip())[:4000]
    res = ""
    try:
        writer = BaseAgent("You write short, realistic client profiles for counseling role-play.", model=model, temperature=0.0)
        res = writer.generate([{"role": "user", "content": (
            f"Topic: {scenario.get('topic', '')}\n"
            f"Background: {bg}\n"
            f"Things this client actually said in a real counseling session:\n{said or '(none)'}\n\n"
            "Write a profile of this client for an actor who will play them: who they are (age range, life situation), what is "
            "going on and since when, the specific events, people and places involved, their feelings, typical thoughts and "
            "beliefs about themselves, what they do or avoid because of it, how it affects relationships and daily life, what "
            "they hope for, and how they talk. Keep every detail consistent with what they said and fill gaps with plausible, "
            "ordinary details. Write it in the third person, 150-220 words, plain prose. Do not mention therapy types, "
            "techniques or advice."
        )}], max_tokens=600)
        res = res.strip()
    except Exception:
        res = ""
    if len(res.split()) < 30:
        res = bg
    _personas[sid] = res
    return res


def get_opening(scenario: Dict[str, Any], model: Optional[str] = None) -> str:
    # the dataset sessions are mostly follow-up sessions, so the first client line is often a fragment
    # from the middle of a talk ("Yeah, I guess I did."). write a real opening from the background instead.
    # cached per scenario so baseline and psyparse start with the exact same words
    sid = scenario.get("dialog_id", "")
    if sid in _openings:
        return _openings[sid]
    first = next((str(t["content"]).strip() for t in scenario.get("dialog", []) if is_client(t) and str(t.get("content", "")).strip()), "")
    res = ""
    try:
        writer = BaseAgent("You write the first thing a counseling client says.", model=model, temperature=0.0)
        res = writer.generate([{"role": "user", "content": (
            f"Client profile: {get_persona(scenario, model)}\n"
            f"Topic: {scenario.get('topic', '')}\n\n"
            "Write what this client says at the start of a counseling session to explain what is going on: first person, "
            "2-3 natural sentences about their own problem, specific to the background, no greeting formulas, "
            "no mention of therapy techniques. Output only the client's words."
        )}])
        # small models like to prefix a label
        res = re.sub(r"^\s*\**(client|patient)\**\s*:\s*", "", res.strip(), flags=re.IGNORECASE).strip().strip('"').strip()
    except Exception:
        res = ""
    if len(res.split()) < 5:
        res = first if len(first.split()) >= 8 else f"Hi. I've been struggling with {str(scenario.get('topic', 'some things')).lower()} lately and I don't know what to do."
    _openings[sid] = res
    return res


class PatientAgent(BaseAgent):
    """
    Patient Agent representing an individual exhibiting specified clinical symptoms.
    Guards against premature termination by preventing early [SESSION_END] leaks.
    """
    def __init__(
        self,
        scenario: Dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.7,
        min_turns: int = 6,
    ):
        self.scenario = scenario
        self.topic = scenario.get("topic", "Mental Health Difficulties")
        self.background = get_persona(scenario, model)
        self.min_turns = min_turns
        self.current_turn = 0

        system_prompt = (
            f"You are roleplaying as a real, human counseling client experiencing {self.topic}.\n"
            f"Who you are (stay consistent with this, it is your life): {self.background}\n\n"
            "Guidelines:\n"
            "- You are the CLIENT, never the counselor. Speak in first person, 2-5 sentences, like a real person talking.\n"
            "- Speak naturally, emotionally, and authentically from your personal background.\n"
            "- Do not use clinical jargon, diagnose yourself, or offer therapeutic solutions.\n"
            "- Express your authentic struggles, hesitations, emotions, and thoughts.\n"
            "- React honestly to the counselor: open up when you feel understood, push back or withdraw when a reply feels generic, rushed or dismissive.\n"
            "- Do not resolve your problems immediately. Building therapeutic progress takes time.\n"
            "- Only if you feel genuine, deep resolution and the counselor has properly guided you through an entire session, "
            "append the token [SESSION_END] at the very end of your response."
        )
        super().__init__(system_prompt=system_prompt, model=model, temperature=temperature)

    def respond(self, therapist_message: str) -> str:
        self.current_turn += 1
        raw_response = self.send(therapist_message)

        if "[SESSION_END]" in raw_response:
            if self.current_turn < self.min_turns:
                cleaned = raw_response.replace("[SESSION_END]", "").strip()
                if self.history and self.history[-1]["role"] == "assistant":
                    self.history[-1]["content"] = cleaned
                return cleaned

        return raw_response

    def simulate_reaction(self, proposed_response: str) -> str:
        """
        Stateless lookahead simulation: predicts the patient's reaction
        to a prospective therapist response without modifying the active history.
        """
        temp_messages = self.get_history() + [{"role": "user", "content": proposed_response}]
        reaction = self.generate(
            messages=temp_messages,
            temperature=self.temperature,
            ephemeral_system=self.system_prompt + "\n[Predict your authentic, immediate reaction to the counselor's utterance.]"
        )
        return reaction.replace("[SESSION_END]", "").strip()

import re
from typing import Dict, Any, Optional
from psyparse.agents.base_agent import BaseAgent

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
        self.background = scenario.get("background", "Experiencing emotional and psychological distress.")
        self.min_turns = min_turns
        self.current_turn = 0

        system_prompt = (
            f"You are roleplaying as a real, human counseling client experiencing {self.topic}.\n"
            f"Background Context: {self.background}\n\n"
            "Guidelines:\n"
            "- Speak naturally, emotionally, and authentically from your personal background.\n"
            "- Do not use clinical jargon, diagnose yourself, or offer therapeutic solutions.\n"
            "- Express your authentic struggles, hesitations, emotions, and thoughts.\n"
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
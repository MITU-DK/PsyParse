from .base_agent import BaseAgent

# seed prompt uses topic, background, opening_turn
# do NOT use theme - it leaks the therapy label (critical leakage fix)
_SEED_TMPL = (
    "You are a person experiencing {topic}. "
    "Your background: {background}. "
    "Opening turn: {opening_turn}. "
    "Respond naturally as this person would. "
    "If you feel the session has reached a natural conclusion and your issue is "
    "resolved, append the exact string [SESSION_END] to your response."
)

# reinforcement reminder - injected per turn, never stored in history
_REMINDER_TMPL = "Reminder: Maintain your persona based on {topic}."


class PatientAgent(BaseAgent):
    def __init__(self, scenario):
        # scenario is one session from dev_scenarios.json / eval_scenarios.json
        topic = str(scenario.get("topic") or "").strip()
        background = str(scenario.get("background") or "").strip()

        # first client utterance from dialog
        opening_turn = ""
        for turn in scenario.get("dialog", []):
            sp = str(turn.get("speaker", "")).lower()
            # skip therapist turns
            if any(h in sp for h in ("therap", "counsel", "doctor", "assistant", "supporter")):
                continue
            if any(h in sp for h in ("client", "patient", "visitor", "seeker", "user")):
                opening_turn = str(turn.get("content") or "").strip()
                if opening_turn:
                    break

        sys_prompt = _SEED_TMPL.format(
            topic=topic,
            background=background,
            opening_turn=opening_turn,
        )
        # temperature policy: patient = 0.7
        super().__init__(sys_prompt, temp=0.7)
        self.topic = topic

    def send(self, therapist_msg, add_reminder=True):
        # inject reminder into user message, do NOT store it in history
        if add_reminder and self.topic:
            reminder = _REMINDER_TMPL.format(topic=self.topic)
            msg_with_reminder = f"{therapist_msg}\n\n{reminder}"
        else:
            msg_with_reminder = therapist_msg

        # build messages manually so we can inject reminder without polluting history
        msgs = self._build_messages(msg_with_reminder)
        reply = self._call_api(msgs, self.temp)

        # store only the raw therapist message (no reminder) in history
        self.history.append({"role": "user", "content": therapist_msg})
        self.history.append({"role": "assistant", "content": reply})
        return reply

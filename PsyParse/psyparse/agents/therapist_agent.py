from .base_agent import BaseAgent

# ---- prompt templates ----

# 5-area question matrix: emotional triggers, cognitive distortions, behavioral patterns,
# interpersonal issues, therapeutic goals
_QUESTION_MATRIX = (
    "1. What situations or events tend to trigger or worsen how you've been feeling?\n"
    "2. When you're feeling this way, what kinds of thoughts tend to run through your mind?\n"
    "3. How has this been affecting your daily routines and activities?\n"
    "4. How has this been impacting your relationships with others?\n"
    "5. What would you like to feel or be able to do differently by the end of our work together?"
)

# stage 1 interview mode
_INTERVIEW_SYS = (
    "You are a compassionate psychological counselor conducting an intake interview. "
    "Ask EXACTLY ONE  question to understand the patient's {target_q}. "
    "Do NOT ask multiple questions in the same message. "
    "Be warm and conversational, not clinical."
)

# stage 2 guidance synthesis (one-shot, not stateful)
_SYNTHESIS_SYS = (
    "You are a psychological counselor. Given these retrieved therapies, output a structured "
    "framework: {\"frameworks\": [{\"therapy\": \"...\", \"techniques\": [...], "
    "\"procedural_steps\": [...]}]}. "
    "You must rely on your internal clinical knowledge to actively generate step-by-step "
    "procedural_steps for how to apply these techniques. "
    "You MUST integrate techniques from at least two distinct therapies from the provided list "
    "into your final framework. Do not rely solely on the first therapy. "
    "You must respond ONLY with a valid JSON object, with no conversational text or markdown wrappers."
)

_ROLLOUT_SYS = (
    "You are a psychological counselor. Using {therapy_type} techniques, specifically "
    "{techniques}, following these procedural steps: {procedural_steps}, and keeping in "
    "mind the patient's profile: {patient_profile}, respond to the patient. "
    "Keep your response conversational, empathetic, and concise (1-3 sentences)."
)

# stage 3b pruning - same template as rollout but called at T_single=0.7
_PRUNING_SYS = _ROLLOUT_SYS

# baseline counselor - no RAG, no therapy label
_BASELINE_SYS = (
    "You are a psychological counselor utilizing standard evidence-based counseling "
    "techniques. The patient is dealing with {topic}. Their background is: {background}. "
    "Provide empathetic and helpful responses. Keep your response conversational and concise (1-3 sentences)."
)

# T_single for candidate generation (stage 3b)
_T_SINGLE = 0.7


class TherapistAgent(BaseAgent):
    def __init__(self):
        # start in interview mode with temp=0.2
        sys_prompt = _INTERVIEW_SYS.format(target_q="emotional triggers")
        super().__init__(sys_prompt, temp=0.2)

    # ----- stage 1: interview -----
    # caller updates target_q each turn to cycle through the 5 question areas

    def send_interview(self, patient_msg, target_q):
        self.swap_prompt(_INTERVIEW_SYS.format(target_q=target_q))
        return self.send(patient_msg)

    # ----- stage 2: guidance synthesis -----
    # stateless - does not touch history

    def synthesize_guidance(self, therapies):
        # therapies: list of dicts with therapy_type, techniques, applicable_conditions
        therapy_text = "\n".join(
            f"- {t['therapy_type']}: techniques={t.get('techniques', [])}, "
            f"conditions={t.get('applicable_conditions', [])}"
            for t in therapies
        )
        msgs = [
            {"role": "system", "content": _SYNTHESIS_SYS},
            {"role": "user", "content": therapy_text},
        ]
        # temp=0.0 not specified for synthesis explicitly, use interview temp
        return self.generate_from(msgs, temp=self.temp, max_tokens=2000)

    # ----- stage 3a: rollout -----
    # receives ONLY one therapy slice - do not pass full guidance framework
    # critical: prevents blending of all three therapies

    def send_rollout(self, patient_msg, therapy_slice, patient_profile):
        sys = _ROLLOUT_SYS.format(
            therapy_type=therapy_slice.get("therapy", ""),
            techniques=", ".join(therapy_slice.get("techniques", [])),
            procedural_steps="; ".join(therapy_slice.get("procedural_steps", [])),
            patient_profile=patient_profile,
        )
        self.swap_prompt(sys)
        # temp stays 0.2 for rollout
        return self.send(patient_msg)

    # ----- stage 3b: pruning candidate generation -----
    # generates n=4 diverse candidates at T_single, stateless

    def generate_candidates(self, patient_msg, therapy_slice, patient_profile, n=4):
        sys = _PRUNING_SYS.format(
            therapy_type=therapy_slice.get("therapy", ""),
            techniques=", ".join(therapy_slice.get("techniques", [])),
            procedural_steps="; ".join(therapy_slice.get("procedural_steps", [])),
            patient_profile=patient_profile,
        )
        self.swap_prompt(sys)
        # sequential loop instead of parallel threads - prevents TPM spike on Groq free tier
        # ponytail: switch back to generate(n=4) if using a paid API with high TPM limits
        return [self.generate(patient_msg, n=1, temp=_T_SINGLE)[0] for _ in range(n)]

    # ----- baseline mode (no RAG) -----

    def setup_baseline(self, topic, background):
        sys = _BASELINE_SYS.format(topic=topic, background=background)
        self.swap_prompt(sys)
        self.history = []

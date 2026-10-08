import re
import json
import logging
from typing import List, Dict, Any, Optional
from psyparse.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

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
        temperature: float = 0.2,
    ):
        system_prompt = self._get_interview_prompt() if mode == "interview" else "You are an expert psychological counselor."
        super().__init__(system_prompt=system_prompt, model=model, temperature=temperature)

    def _get_interview_prompt(self) -> str:
        return (
            "You are a professional psychological counselor conducting an initial structured intake assessment. "
            "Your objective is to explore the patient's core problems, emotional states, symptoms, cognitive distortions, "
            "interpersonal issues, and therapeutic goals with warmth, validation, and active listening. "
            "Ask exactly ONE clear, empathetic, and open-ended target question per turn."
        )

    def conduct_interview_turn(self, target_question: str, patient_last_response: Optional[str] = None) -> str:
        instruction = (
            f"Target assessment focus for this turn: '{target_question}'.\n"
            "Acknowledge and empathize with what the patient just shared, then naturally incorporate the target question. "
            "Do NOT ask multiple questions. Focus solely on this single question."
        )
        if patient_last_response:
            return self.send(f"{patient_last_response}\n\n[Instruction: {instruction}]")
        return self.send(f"[Instruction: {instruction}]")

    def synthesize_guidance(self, top_therapies: List[Dict[str, Any]], patient_profile: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            "You are a master clinical supervisor. Synthesize a unified, multi-therapy counseling guidance framework "
            "tailored to the following patient profile:\n"
            f"{json.dumps(patient_profile, indent=2)}\n\n"
            f"Candidate therapies retrieved: {json.dumps(top_therapies, indent=2)}\n\n"
            "Requirements:\n"
            "1. You MUST integrate techniques from at least two distinct therapies.\n"
            "2. Provide explicit step-by-step 'procedural_steps' for the therapist to apply.\n"
            "3. Output strictly valid JSON with this exact schema:\n"
            "{\n"
            '  "frameworks": [\n'
            '    {\n'
            '      "therapy": "Therapy Name",\n'
            '      "techniques": ["Technique 1", "Technique 2"],\n'
            '      "procedural_steps": ["Step 1: ...", "Step 2: ..."]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Respond ONLY with valid JSON."
        )
        raw = self.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        try:
            cleaned = self._clean_json(raw)
            data = json.loads(cleaned)
            if "frameworks" in data and len(data["frameworks"]) >= 1:
                return data
        except Exception as e:
            logger.error("Failed to parse guidance JSON: %s. Using deterministic fallback.", e)

        return {
            "frameworks": [
                {
                    "therapy": th.get("therapy_type", "CBT"),
                    "techniques": th.get("techniques", ["Cognitive Restructuring"]),
                    "procedural_steps": [
                        "Validate the client's emotional distress",
                        "Explore underlying belief patterns",
                        "Collaboratively identify actionable coping mechanisms"
                    ]
                }
                for th in top_therapies[:2]
            ]
        }

    def setup_rollout_mode(self, therapy_slice: Dict[str, Any], patient_profile: Dict[str, Any]) -> None:
        th_name = therapy_slice.get("therapy", "CBT")
        techs = ", ".join(therapy_slice.get("techniques", []))
        steps = " -> ".join(therapy_slice.get("procedural_steps", []))
        guided_prompt = (
            f"You are a professional counselor conducting an active therapy session using {th_name}.\n"
            f"Target Techniques: {techs}\n"
            f"Procedural Plan: {steps}\n"
            f"Patient Context: {json.dumps(patient_profile)}\n\n"
            "Apply these therapy techniques explicitly and concretely. Validate emotional distress, "
            "maintain steady therapeutic direction, and avoid generic conversational small talk."
        )
        self.swap_system_prompt(guided_prompt)

    def generate_diverse_branches(
        self,
        therapy_slice: Dict[str, Any],
        patient_profile: Dict[str, Any],
        n: int = 4,
        temperature: float = 0.7,
    ) -> List[str]:
        # CRITICAL FIX: Force semantic diversity across the 4 branches
        clinical_angles = [
            "Cognitive Angle: Challenge a distorted thought or reframe a perspective.",
            "Emotional Angle: Provide deep, compassionate validation of the underlying affect.",
            "Behavioral Angle: Propose a concrete action, grounding exercise, or experiment.",
            "Socratic Angle: Ask a penetrating, reflective question to foster insight."
        ]
        history_msgs = self.get_history()
        branches = []

        for i in range(n):
            strategy_directive = clinical_angles[i % len(clinical_angles)]
            prompt = (
                f"[Mandatory Clinical Angle: {strategy_directive}]\n"
                f"Using {therapy_slice.get('therapy', 'CBT')}, generate the next counselor utterance. "
                "Output ONLY the counselor's direct spoken response."
            )
            resp = self.generate(
                messages=history_msgs + [{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            cleaned = resp.replace("Counselor:", "").replace("Therapist:", "").strip()
            branches.append(cleaned)

        return branches

    @staticmethod
    def _clean_json(text: str) -> str:
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else text
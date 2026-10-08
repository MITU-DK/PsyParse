import os
import re
import json
import logging
from typing import Dict, Any, List
from psyparse.retrieval.hybrid_search import HybridRetriever
from psyparse.pipeline.run_full import run_psyparse_pipeline
from psyparse.agents.base_agent import BaseAgent
from psyparse.agents.patient_agent import PatientAgent

logger = logging.getLogger(__name__)

class EvaluatorHarness:
    def __init__(
        self,
        retriever: HybridRetriever,
        model: str = "qwen2.5:14b",
    ):
        self.retriever = retriever
        self.model = model
        self.judge_prompt_template = self._get_strict_judge_prompt()
        self.judge_agent = BaseAgent(system_prompt="You are a strict, expert clinical psychology evaluator.", model=model, temperature=0.0)

    def _get_strict_judge_prompt(self) -> str:
        return (
            "Evaluate this therapeutic transcript on 6 dimensions using a strict 1.0 to 10.0 scale (use one decimal, e.g., 7.4).\n"
            "Do NOT give scores above 8.0 unless the therapist demonstrates expert, explicit clinical methodology.\n\n"
            "1. coherence: Logical flow and topical transitions.\n"
            "2. completeness: Depth of exploring underlying trauma/symptoms (1-4: superficial, 8-10: comprehensive).\n"
            "3. humaneness: Anticipatory empathy and authentic warmth.\n"
            "4. technique: Explicit execution of evidence-based techniques (1-5: generic advice, 8-10: explicit CBT/ACT/SFBT exercises).\n"
            "5. structure: Goal progression and phased therapeutic advancement.\n"
            "6. context_retention: Deep integration of the patient's unique background.\n\n"
            "Transcript:\n{TRANSCRIPT}\n\n"
            "Output strictly valid JSON:\n"
            "{\"coherence\": 0.0, \"completeness\": 0.0, \"humaneness\": 0.0, \"technique\": 0.0, \"structure\": 0.0, \"context_retention\": 0.0}"
        )

    def run_baseline_dialogue(self, scenario: Dict[str, Any], max_turns: int = 10) -> List[Dict[str, str]]:
        topic = scenario.get("topic", "General")
        background = scenario.get("background", "")
        therapist = BaseAgent(
            system_prompt=f"You are a counselor. The patient is experiencing {topic}. Background: {background}. Provide standard empathetic responses.",
            model=self.model,
            temperature=0.7,
        )
        patient = PatientAgent(scenario=scenario, model=self.model, temperature=0.7, min_turns=8)
        opening = scenario.get("opening_turn", "I've been struggling lately.")
        patient.add_message("assistant", opening)

        therapist_msg = therapist.send(f"Patient: {opening}")
        patient.respond(therapist_msg)

        for _ in range(max_turns - 1):
            therapist_msg = therapist.send(patient.get_history()[-1]["content"])
            patient_resp = patient.respond(therapist_msg)
            if "[SESSION_END]" in patient_resp:
                break
        return patient.get_history()

    def evaluate_transcript(self, transcript: List[Dict[str, str]]) -> Dict[str, float]:
        transcript_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in transcript])
        prompt = self.judge_prompt_template.replace("{TRANSCRIPT}", transcript_str)

        raw = self.judge_agent.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        try:
            cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE)
            data = json.loads(cleaned)
            # Scale the 1.0-10.0 score up to 10-100 for final reporting
            return {k: round(float(data.get(k, 5.0)) * 10.0, 2) for k in [
                "coherence", "completeness", "humaneness", "technique", "structure", "context_retention"
            ]}
        except Exception as e:
            logger.error(f"Judge parsing failed: {e}")
            return {k: 50.0 for k in ["coherence", "completeness", "humaneness", "technique", "structure", "context_retention"]}

    def evaluate_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        baseline_transcript = self.run_baseline_dialogue(scenario)
        psyparse_result = run_psyparse_pipeline(scenario, self.retriever, model=self.model)

        baseline_scores = self.evaluate_transcript(baseline_transcript)
        psyparse_scores = self.evaluate_transcript(psyparse_result["transcript"])

        return {
            "scenario_id": scenario.get("dialog_id", "unknown"),
            "topic": scenario.get("topic", "General"),
            "baseline_scores": baseline_scores,
            "psyparse_scores": psyparse_scores,
            "delta": {k: round(psyparse_scores[k] - baseline_scores[k], 2) for k in baseline_scores},
            "best_therapy": psyparse_result["best_therapy"]
        }
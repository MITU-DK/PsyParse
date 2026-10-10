import json
import os
import re
from psyparse.agents.base_agent import BaseAgent

prompt_template = (
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

judge = BaseAgent("You are a strict, expert clinical psychology evaluator.", temperature=0.0)

t1 = "Therapist: Tell me about your day.\nPatient: It was bad."
t2 = "Therapist: Let's do a cognitive restructuring exercise. What is the automatic thought?\nPatient: I am worthless."

r1 = judge.generate([{"role": "user", "content": prompt_template.replace("{TRANSCRIPT}", t1)}], response_format={"type":"json_object"})
r2 = judge.generate([{"role": "user", "content": prompt_template.replace("{TRANSCRIPT}", t2)}], response_format={"type":"json_object"})

print("T1:", r1)
print("T2:", r2)

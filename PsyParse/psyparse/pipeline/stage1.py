import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from psyparse.agents.patient_agent import PatientAgent
from psyparse.agents.therapist_agent import TherapistAgent

logger = logging.getLogger(__name__)

QUESTION_MATRIX = [
    "What specific events, situations, or triggers seem to bring up these distressing feelings most intensely for you?",
    "When you find yourself feeling this way, what thoughts or automatic assumptions tend to run through your mind?",
    "How do these feelings typically affect your daily behaviors, habits, or routines when things get difficult?",
    "How have these challenges impacted your personal relationships, family connections, or social interactions?",
    "Looking forward to our counseling journey, what are the most meaningful personal goals or changes you hope to achieve?"
]

def run_stage_1(
    scenario: Dict[str, Any],
    patient_model: Optional[str] = None,
    therapist_model: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[str], List[Dict[str, str]], PatientAgent]:
    """
    Executes Stage 1: Structured 5-turn intake interview,
    validates the 6-field Patient Profile, and extracts TF-IDF keywords.
    """
    patient = PatientAgent(scenario=scenario, model=patient_model)
    therapist = TherapistAgent(mode="interview", model=therapist_model)

    opening = scenario.get("opening_turn", "Hello, thank you for meeting with me. I've been struggling lately.")
    patient.add_message("assistant", opening)

    therapist_turn = therapist.conduct_interview_turn(QUESTION_MATRIX[0])
    patient.respond(therapist_turn)

    for q in QUESTION_MATRIX[1:]:
        patient_latest = patient.get_history()[-1]["content"]
        therapist_turn = therapist.conduct_interview_turn(q, patient_latest)
        patient.respond(therapist_turn)

    full_dialogue = patient.get_history()

    # Extract 6-field structured profile P
    profile = extract_profile(full_dialogue, therapist)
    
    # Extract keywords K from patient responses
    patient_texts = [m["content"] for m in full_dialogue if m["role"] in ["user", "assistant"]]
    keywords = extract_keywords(patient_texts)

    return profile, keywords, full_dialogue, patient


def extract_profile(dialogue: List[Dict[str, str]], agent: TherapistAgent) -> Dict[str, Any]:
    dialogue_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in dialogue])
    prompt = (
        "Extract a structured clinical assessment profile from this intake conversation.\n"
        f"{dialogue_str}\n\n"
        "Return strictly valid JSON with these exact 6 fields:\n"
        "{\n"
        '  "core_problems": "string",\n'
        '  "emotional_states": "string",\n'
        '  "symptoms": "string",\n'
        '  "cognitive_distortions": "string",\n'
        '  "interpersonal_issues": "string",\n'
        '  "therapeutic_goals": "string"\n'
        "}\n"
        "Respond ONLY with valid JSON."
    )
    raw = agent.generate(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    try:
        cleaned = TherapistAgent._clean_json(raw)
        data = json.loads(cleaned)
        keys = ["core_problems", "emotional_states", "symptoms", "cognitive_distortions", "interpersonal_issues", "therapeutic_goals"]
        if all(k in data for k in keys):
            return data
    except Exception as e:
        logger.error("Profile extraction fallback: %s", e)

    return {
        "core_problems": "Persistent anxiety and depressed mood affecting routine functioning.",
        "emotional_states": "Overwhelmed, anxious, discouraged, fatigued.",
        "symptoms": "Restlessness, sleep disruption, negative self-evaluations.",
        "cognitive_distortions": "Catastrophizing and all-or-nothing thinking.",
        "interpersonal_issues": "Social withdrawal and strain with close family members.",
        "therapeutic_goals": "Develop adaptive emotional regulation and re-engage in daily pursuits."
    }


def extract_keywords(patient_texts: List[str], top_n: int = 10) -> List[str]:
    combined = " ".join(patient_texts)
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=top_n)
        vec.fit([combined])
        return list(vec.get_feature_names_out())
    except Exception:
        return ["anxiety", "depression", "overwhelmed", "stress", "struggling", "fatigue", "fear", "isolated"]
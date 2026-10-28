import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import config
from psyparse.agents.base_agent import BaseAgent, format_dialogue
from psyparse.agents.patient_agent import PatientAgent, get_opening
from psyparse.agents.therapist_agent import TherapistAgent
from psyparse.retrieval.hybrid_search import HybridRetriever

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
    retriever: Optional[HybridRetriever] = None,
) -> Tuple[Dict[str, Any], List[str], List[Dict[str, str]], PatientAgent]:
    """
    Executes Stage 1: Structured 5-turn intake interview,
    validates the 6-field Patient Profile, and extracts TF-IDF keywords.
    """
    patient = PatientAgent(scenario=scenario, model=patient_model)
    therapist = TherapistAgent(mode="interview", model=therapist_model)
    # spec: interview prompt = baseline counseling strategies + question matrix
    qs = "\n".join(f"- {q}" for q in QUESTION_MATRIX[:config.INTAKE_TURNS])
    therapist.swap_system_prompt(f"{config.COUNSELOR_PROMPT}\nIn this first part of the session, counsel the client as you normally would "
                                 f"and also ask the following questions systematically, one per reply:\n{qs}")

    opening = get_opening(scenario, model=patient_model)
    patient.add_message("assistant", opening)

    # therapist has to actually hear the opening, otherwise turn 1 ignores it
    patient_latest = opening
    for q in QUESTION_MATRIX[:config.INTAKE_TURNS]:
        therapist_turn = therapist.conduct_interview_turn(q, patient_latest)
        patient_latest = patient.respond(therapist_turn).replace("[SESSION_END]", "").strip()

    full_dialogue = patient.get_history()

    # Extract 6-field structured profile P
    profile = extract_profile(full_dialogue, therapist_model)

    # Extract keywords K from the patient's own words only
    patient_texts = [m["content"] for m in full_dialogue if m["role"] == "assistant"]
    keywords = extract_keywords(patient_texts, retriever=retriever)

    return profile, keywords, full_dialogue, patient


def extract_profile(dialogue: List[Dict[str, str]], model: Optional[str] = None) -> Dict[str, Any]:
    # own agent, the interview therapist's system prompt says "talk to the client, no lists" which fights json mode
    agent = BaseAgent("You are a clinical assessment assistant. You output only valid JSON.", model=model, temperature=0.0)
    dialogue_str = format_dialogue(dialogue)
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
    keys = ["core_problems", "emotional_states", "symptoms", "cognitive_distortions", "interpersonal_issues", "therapeutic_goals"]
    for attempt in range(2):
        raw = agent.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0 if attempt == 0 else 0.3,
            response_format={"type": "json_object"},
            max_tokens=1000,
        )
        try:
            data = json.loads(TherapistAgent._clean_json(raw))
            if all(k in data for k in keys):
                return data
        except Exception as e:
            logger.error("Profile extraction failed (attempt %d): %s", attempt + 1, e)

    print("  [warn] profile extraction failed, using a generic profile (retrieval will be weak for this scenario)")
    return {
        "core_problems": "Persistent anxiety and depressed mood affecting routine functioning.",
        "emotional_states": "Overwhelmed, anxious, discouraged, fatigued.",
        "symptoms": "Restlessness, sleep disruption, negative self-evaluations.",
        "cognitive_distortions": "Catastrophizing and all-or-nothing thinking.",
        "interpersonal_issues": "Social withdrawal and strain with close family members.",
        "therapeutic_goals": "Develop adaptive emotional regulation and re-engage in daily pursuits."
    }


def extract_keywords(patient_texts: List[str], top_n: int = 10, retriever: Optional[HybridRetriever] = None) -> List[str]:
    try:
        if retriever is not None:
            kw = retriever.keywords(patient_texts, top_n=top_n)
            if kw:
                return kw
        # no db idf available, fall back to plain term weights
        vec = TfidfVectorizer(stop_words="english", max_features=top_n)
        vec.fit([" ".join(patient_texts)])
        return list(vec.get_feature_names_out())
    except Exception:
        return ["anxiety", "depression", "overwhelmed", "stress", "struggling", "fatigue", "fear", "isolated"]

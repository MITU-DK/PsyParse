import pickle
import re
from pathlib import Path

import numpy as np

from ..agents.evaluation_agent import EvaluationAgent
from ..agents.patient_agent import PatientAgent
from ..agents.therapist_agent import TherapistAgent

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# 5 question areas in order - used to feed one target question per turn
_QUESTION_AREAS = [
    "emotional triggers",
    "cognitive distortions",
    "behavioral patterns",
    # "interpersonal issues",  # removed for testing - restore before submission
    # "therapeutic goals",     # removed for testing - restore before submission
]

# required 6-field keys
_PROFILE_KEYS = {
    "core_problems",
    "emotional_states",
    "symptoms",
    "cognitive_distortions",
    "interpersonal_issues",
    "therapeutic_goals",
}

# how many top tfidf keywords to extract
_TOP_K_KEYWORDS = 10


def _check_single_question(therapist_reply):
    # count question marks to check therapist didnt dump multiple questions in one turn
    return therapist_reply.count("?") <= 1


def _extract_keywords(patient_texts, tfidf):
    combined = " ".join(patient_texts)
    vec = tfidf.transform([combined])
    vocab = tfidf.get_feature_names_out()
    scores = np.asarray(vec.todense()).flatten()
    top_idx = scores.argsort()[::-1][:_TOP_K_KEYWORDS]
    keywords = [vocab[i] for i in top_idx if scores[i] > 0]
    return keywords


def run_stage1(scenario):
    patient = PatientAgent(scenario)
    therapist = TherapistAgent()
    eval_agent = EvaluationAgent()

    # fix pickle loading by making tokenize available in __main__
    import sys
    from ..retrieval.hybrid_search import tokenize
    sys.modules["__main__"].tokenize = tokenize

    # load tfidf fitted in phase 1
    with open(DATA_DIR / "tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    patient_texts = []  # collect patient-only utterances for keyword extraction

    # one question area per turn, in order
    for area in _QUESTION_AREAS:
        # (a) therapist generates natural turn incorporating the target question
        max_q_retries = 3
        t_reply = None
        t_temp_bump = 0.0
        for attempt in range(max_q_retries):
            # temporarily bump temp if previous attempt failed the check
            orig_temp = therapist.temp
            therapist.temp = orig_temp + t_temp_bump
            t_reply = therapist.send_interview(
                # if first turn, seed with a greeting; else use last patient utterance
                patient.history[-1]["content"] if patient.history else "Hello.",
                target_q=area,
            )
            therapist.temp = orig_temp  # restore
            if _check_single_question(t_reply):
                break
            # retry: pop the bad turn from therapist history, bump temp
            therapist.history = therapist.history[:-2]
            t_temp_bump += 0.2
        else:
            # all retries failed - flag but continue with last reply
            print(f"[warn] therapist asked multiple questions on area '{area}' after retries")

        # (b) patient responds (reminder injected inside PatientAgent.send)
        p_reply = patient.send(t_reply)
        patient_texts.append(p_reply)

        if "[SESSION_END]" in p_reply:
            # session ended early - stop interview
            print("[info] patient signaled SESSION_END during interview")
            break

    # extract structured profile and keywords
    # use full conversation history for profile extraction
    conversation = therapist.get_history()

    profile = None
    for attempt in range(3):
        raw_profile = eval_agent.extract_profile(conversation)
        # validate exactly 6 fields
        if isinstance(raw_profile, dict) and _PROFILE_KEYS.issubset(raw_profile.keys()):
            profile = raw_profile
            break
        print(f"[warn] profile extraction attempt {attempt+1} missing fields, retrying...")

    if profile is None:
        # flag for manual review after 3 retries
        print("[error] profile extraction failed after 3 retries - flagged for manual review")
        profile = {k: "" for k in _PROFILE_KEYS}

    # restrict to 3-field subset for embedding (shared with DB case profiles)
    # full 6-field profile is preserved separately for downstream stages
    embed_profile = {
        "core_problems": profile.get("core_problems", ""),
        "emotional_states": profile.get("emotional_states", ""),
        "symptoms": profile.get("symptoms", ""),
    }

    # keywords from patient responses only via tfidf
    keywords = _extract_keywords(patient_texts, tfidf)

    return {
        "profile": profile,          # full 6-field (for stage 3a/3b context)
        "embed_profile": embed_profile,  # 3-field (for FAISS query in stage 2)
        "keywords": keywords,
        "interview_history": therapist.get_history(),
    }

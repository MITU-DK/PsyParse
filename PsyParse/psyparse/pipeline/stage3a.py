import logging
from typing import Dict, Any, List, Tuple
from psyparse.agents.patient_agent import PatientAgent
from psyparse.agents.therapist_agent import TherapistAgent
from psyparse.agents.evaluation_agent import EvaluationAgent

logger = logging.getLogger(__name__)

def run_stage_3a(
    selected_therapies: List[Dict[str, Any]],
    guidance_framework: Dict[str, Any],
    patient_profile: Dict[str, Any],
    patient: PatientAgent,
    evaluator: EvaluationAgent,
    num_rounds: int = 3,
    tau_resp: float = 2.5,
    max_retries: int = 2,
    we_m: float = 0.5,
    wt_m: float = 0.5,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """
    Stage 3a: Multi-Turn Rollout for Personalized Therapy Selection.
    Simulates prospective trajectories for each top candidate and selects T*.
    Guarantees that best_therapy is never null.
    """
    post_interview_snapshot = patient.snapshot()
    frameworks = guidance_framework.get("frameworks", [])

    framework_lookup = {f.get("therapy"): f for f in frameworks}
    trajectory_scores: Dict[str, float] = {}
    rollout_logs: Dict[str, Any] = {}

    for cand in selected_therapies:
        th_type = cand.get("therapy_type", "CBT")
        th_slice = framework_lookup.get(th_type, {
            "therapy": th_type,
            "techniques": cand.get("techniques", ["Restructuring"]),
            "procedural_steps": ["Assess", "Intervene", "Consolidate"]
        })

        therapist = TherapistAgent(mode="guided", model=evaluator.model)
        therapist.setup_rollout_mode(th_slice, patient_profile)
        patient.restore(post_interview_snapshot)

        cum_score = 0.0
        convo_history_str = "Initial assessment completed."
        prev_reaction = ""

        for r in range(1, num_rounds + 1):
            therapist_msg = ""
            best_turn_score = -1.0
            best_attempt_msg = ""

            if r == 1:
                prompt_msg = f"Patient is waiting for active guidance using {th_type}. Provide a concise counseling response."
            else:
                prompt_msg = f"Patient said: '{prev_reaction}'.\nContinue providing active guidance using {th_type}. Provide a concise counseling response."

            for retry in range(max_retries + 1):
                temp = 0.2 if retry == 0 else 0.5
                candidate_resp_list = therapist.generate(prompt=prompt_msg, n=1, temp=temp)
                candidate_resp = candidate_resp_list[0] if isinstance(candidate_resp_list, list) else candidate_resp_list
                
                scores = evaluator.score_rollout(convo_history_str, candidate_resp, th_type)
                s_resp = (we_m * scores.get("empathy", 3.0)) + (wt_m * scores.get("alignment", 3.0))

                if s_resp > best_turn_score:
                    best_turn_score = s_resp
                    best_attempt_msg = candidate_resp

                if s_resp >= tau_resp:
                    break

            therapist_msg = best_attempt_msg
            cum_score += best_turn_score

            # Commit to therapist history
            therapist.add_message("user", prompt_msg)
            therapist.add_message("assistant", therapist_msg)

            reaction = patient.respond(therapist_msg)
            prev_reaction = reaction
            convo_history_str += f"\nTherapist: {therapist_msg}\nPatient: {reaction}"

            if "[SESSION_END]" in reaction:
                remaining_rounds = num_rounds - r
                cum_score += remaining_rounds * 5.0
                break

        trajectory_scores[th_type] = cum_score
        rollout_logs[th_type] = {"score": cum_score, "slice": th_slice}

    patient.restore(post_interview_snapshot)

    # Deterministic winner selection (Eq. 3 argmax with Stage 2 fallback)
    if not trajectory_scores or max(trajectory_scores.values()) <= 0.0:
        winning_type = selected_therapies[0]["therapy_type"]
    else:
        winning_type = max(trajectory_scores, key=lambda k: trajectory_scores[k])

    winning_slice = framework_lookup.get(winning_type, {
        "therapy": winning_type,
        "techniques": selected_therapies[0].get("techniques", ["Cognitive Restructuring"]),
        "procedural_steps": ["Validate", "Identify Distortions", "Develop Coping Action"]
    })
    winning_score = trajectory_scores.get(winning_type, selected_therapies[0].get("s_th", 1.0))

    return winning_slice, rollout_logs, winning_score
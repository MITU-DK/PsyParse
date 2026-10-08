import logging
from typing import Dict, Any, List
from psyparse.agents.patient_agent import PatientAgent
from psyparse.agents.therapist_agent import TherapistAgent
from psyparse.agents.evaluation_agent import EvaluationAgent

logger = logging.getLogger(__name__)

def run_stage_3b(
    best_therapy_slice: Dict[str, Any],
    patient_profile: Dict[str, Any],
    real_patient: PatientAgent,
    evaluator: EvaluationAgent,
    max_turns: int = 10,
    min_turns: int = 6,
    num_branches: int = 4,
    t_single: float = 0.7,
    we_s: float = 0.5,
    ws: float = 0.5,
) -> List[Dict[str, str]]:
    """
    Stage 3b: Response Pruning for Empathetic Optimization.
    Generates n diverse candidates, simulates one-step patient reactions,
    prunes via Eq. 4, and prevents premature session closure.
    """
    therapist = TherapistAgent(mode="guided", model=evaluator.model)
    therapist.setup_rollout_mode(best_therapy_slice, patient_profile)

    # Create an independent Simulated Patient synchronized with current history
    simulated_patient = PatientAgent(
        scenario=real_patient.scenario,
        model=evaluator.model,
        temperature=0.7,
        min_turns=min_turns
    )
    simulated_patient.set_history(real_patient.get_history())

    th_name = best_therapy_slice.get("therapy", "Counseling")

    for turn in range(1, max_turns + 1):
        # 1. Generate diverse therapist candidate branches
        candidates = therapist.generate_diverse_branches(
            therapy_slice=best_therapy_slice,
            patient_profile=patient_profile,
            n=num_branches,
            temperature=t_single,
        )

        # 2. Simulate patient reactions & score pairs (Eq. 4)
        best_r = candidates[0]
        best_score = -1.0
        dialogue_history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in real_patient.get_history()])

        for cand in candidates:
            sim_reaction = simulated_patient.simulate_reaction(cand)
            scores = evaluator.score_pruning(dialogue_history_str, cand, sim_reaction)
            s_single = (we_s * scores.get("empathy", 3.0)) + (ws * scores.get("stability", 3.0))

            if s_single > best_score:
                best_score = s_single
                best_r = cand

        # 3. Commit best response to the real dialogue
        therapist.add_message("assistant", best_r)

        # 4. Real patient reacts
        real_reaction = real_patient.respond(best_r)
        therapist.add_message("user", real_reaction)

        # 5. Keep simulated patient timeline synchronized
        simulated_patient.set_history(real_patient.get_history())

        # 6. Guarded session termination check
        if "[SESSION_END]" in real_reaction:
            if turn < min_turns:
                real_reaction = real_reaction.replace("[SESSION_END]", "").strip()
                # Cleanse patient history of the [SESSION_END] token
                if real_patient.history and "[SESSION_END]" in real_patient.history[-1]["content"]:
                    real_patient.history[-1]["content"] = real_patient.history[-1]["content"].replace("[SESSION_END]", "").strip()
                # Inject hidden reinforcement to keep the dialogue going using 'user' role to prevent API crashes
                real_patient.add_message("user", "[Internal Instruction: You still feel unresolved tension. Elaborate on your struggles in your next response.]")
            else:
                break

    return real_patient.get_history()
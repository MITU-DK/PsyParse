import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple
import config
from psyparse.agents.base_agent import format_dialogue, flip_roles
from psyparse.agents.patient_agent import PatientAgent
from psyparse.agents.therapist_agent import TherapistAgent
from psyparse.agents.evaluation_agent import EvaluationAgent

logger = logging.getLogger(__name__)

def run_stage_3b(
    best_therapy_slice: Dict[str, Any],
    patient_profile: Dict[str, Any],
    real_patient: PatientAgent,
    evaluator: EvaluationAgent,
    max_turns: int = config.THERAPY_TURNS,
    min_turns: int = config.MIN_THERAPY_TURNS,
    num_branches: int = config.PRUNING_BRANCHES,
    t_single: float = config.TEMP_SINGLE,
    we_s: float = config.WE_S,
    ws: float = config.WS,
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """
    Stage 3b: Response Pruning for Empathetic Optimization.
    Generates n diverse candidates, simulates one-step patient reactions,
    prunes via Eq. 4, and prevents premature session closure.
    """
    therapist = TherapistAgent(mode="guided", model=evaluator.model)
    therapist.setup_rollout_mode(best_therapy_slice, patient_profile)
    # therapist remembers the intake interview
    therapist.set_history(flip_roles(real_patient.get_history()))

    # count only therapy turns from here, PatientAgent.respond blocks [SESSION_END] before min_turns
    real_patient.current_turn = 0
    real_patient.min_turns = min_turns

    # Create an independent Simulated Patient synchronized with current history
    simulated_patient = PatientAgent(
        scenario=real_patient.scenario,
        model=real_patient.model,
        temperature=0.7,
        min_turns=10 ** 6,
    )
    simulated_patient.set_history(real_patient.get_history())

    logs = []
    for turn in range(1, max_turns + 1):
        # 1. Generate diverse therapist candidate branches
        candidates = therapist.generate_diverse_branches(
            therapy_slice=best_therapy_slice,
            turn=turn,
            max_turns=max_turns,
            n=num_branches,
            temperature=t_single,
        )

        # 2. Simulate patient reactions & score pairs (Eq. 4)
        hist = format_dialogue(real_patient.get_history())

        def score(cand):
            sim_reaction = simulated_patient.simulate_reaction(cand)
            sc = evaluator.score_pruning(hist, cand, sim_reaction)
            return (we_s * sc.get("empathy", 3.0)) + (ws * sc.get("stability", 3.0))

        # lookahead + scoring is stateless per candidate, so run them side by side
        with ThreadPoolExecutor(max_workers=max(1, min(config.PARALLEL_CALLS, len(candidates)))) as ex:
            turn_scores = [round(x, 3) for x in ex.map(score, candidates)]
        best_i = max(range(len(candidates)), key=lambda i: turn_scores[i])  # first one wins ties
        best_r = candidates[best_i]

        # 3. Commit best response, real patient reacts
        therapist.add_message("assistant", best_r)
        real_reaction = real_patient.respond(best_r)
        therapist.add_message("user", real_reaction.replace("[SESSION_END]", "").strip())

        # 4. Keep simulated patient timeline synchronized
        simulated_patient.set_history(real_patient.get_history())
        logs.append({"turn": turn, "branch_scores": turn_scores, "chosen": best_i})
        print(f"  [stage3b] turn {turn}/{max_turns}: branch scores {turn_scores} -> picked {best_i + 1}")

        # 5. patient can only end after min_turns (enforced inside PatientAgent.respond)
        if "[SESSION_END]" in real_reaction:
            break

    return real_patient.get_history(), logs

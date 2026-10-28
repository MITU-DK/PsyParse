import random
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

    # seeded per scenario so a rerun picks the same way on ties / shuffles
    rng = random.Random(str(real_patient.scenario.get("dialog_id", "")))
    logs = []
    for turn in range(1, max_turns + 1):
        # 1. Generate diverse therapist candidate branches
        candidates = therapist.generate_diverse_branches(
            turn=turn,
            max_turns=max_turns,
            n=num_branches,
            temperature=t_single,
        )

        # 2. simulate the patient's reaction to each candidate (lookahead is stateless, so run side by side)
        workers = max(1, min(config.PARALLEL_CALLS, len(candidates)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            reactions = list(ex.map(simulated_patient.simulate_reaction, candidates))

        # 3. Eq. 4 scores, all candidates in one call
        # last few messages are enough context here and keep the prompt small
        hist = format_dialogue(real_patient.get_history()[-8:])
        # shuffled order (spec) and again reversed, then averaged: the scorer tends to just go down the list
        order = list(range(len(candidates)))
        rng.shuffle(order)

        def score(o):
            r = evaluator.rank_candidates(hist, [candidates[i] for i in o], [reactions[i] for i in o])
            if r is None:
                return None
            res = [None] * len(candidates)
            for pos, i in enumerate(o):
                res[i] = r[pos]
            return res
        with ThreadPoolExecutor(max_workers=max(1, min(config.PARALLEL_CALLS, 2))) as ex:
            runs = [x for x in ex.map(score, [order, order[::-1]]) if x]
        if runs:
            scored = [{k: round(sum(r[i][k] for r in runs) / len(runs), 3) for k in ["empathy", "stability"]}
                      for i in range(len(candidates))]
        else:
            full = format_dialogue(real_patient.get_history())
            with ThreadPoolExecutor(max_workers=workers) as ex:
                scored = list(ex.map(lambda cr: evaluator.score_pruning(full, cr[0], cr[1]), zip(candidates, reactions)))
        # Eq. 4: S_single = we_s * E + ws * S
        turn_scores = [round(we_s * sc.get("empathy", 3.0) + ws * sc.get("stability", 3.0), 3) for sc in scored]

        # best Eq. 4 score, random tie break
        top = max(turn_scores)
        best_i = rng.choice([i for i, x in enumerate(turn_scores) if x == top])
        best_r = candidates[best_i]

        # 4. Commit best response, real patient reacts
        therapist.add_message("assistant", best_r)
        real_reaction = real_patient.respond(best_r)
        therapist.add_message("user", real_reaction.replace("[SESSION_END]", "").strip())

        # 5. Keep simulated patient timeline synchronized
        simulated_patient.set_history(real_patient.get_history())
        # keep the candidates so we can see later what the scorer picked and what it threw away
        logs.append({"turn": turn, "branch_scores": turn_scores, "chosen": best_i, "scores": scored,
                     "candidates": candidates, "reactions": reactions})
        print(f"  [stage3b] turn {turn}/{max_turns}: branch scores {turn_scores} -> picked {best_i + 1}")

        # 6. patient can only end after min_turns (enforced inside PatientAgent.respond)
        if "[SESSION_END]" in real_reaction:
            break

    return real_patient.get_history(), logs

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple
import config
from psyparse.agents.base_agent import format_dialogue, flip_roles
from psyparse.agents.patient_agent import PatientAgent
from psyparse.agents.therapist_agent import TherapistAgent, fallback_slice
from psyparse.agents.evaluation_agent import EvaluationAgent

logger = logging.getLogger(__name__)

def run_stage_3a(
    selected_therapies: List[Dict[str, Any]],
    guidance_framework: Dict[str, Any],
    patient_profile: Dict[str, Any],
    patient: PatientAgent,
    evaluator: EvaluationAgent,
    num_rounds: int = config.ROLLOUT_ROUNDS,
    tau_resp: float = config.TAU_RESP,
    max_retries: int = config.ROLLOUT_RETRIES,
    we_m: float = config.WE_M,
    wt_m: float = config.WT_M,
    total_turns: int = config.THERAPY_TURNS,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """
    Stage 3a: Multi-Turn Rollout for Personalized Therapy Selection.
    Simulates prospective trajectories for each top candidate and selects T*.
    The real patient is never touched here, rollouts run on a simulated copy.
    """
    snap = patient.snapshot()
    lookup = {str(f.get("therapy", "")).lower(): f for f in guidance_framework.get("frameworks", [])}

    def rollout(cand):
        th = cand.get("therapy_type", "CBT")
        sl = lookup.get(th.lower()) or fallback_slice(cand)

        therapist = TherapistAgent(mode="guided", model=evaluator.model)
        # only this therapy's slice of the guidance framework (spec)
        therapist.setup_rollout_mode(sl, patient_profile)
        therapist.set_history(flip_roles(snap))

        # simulated patient forked from the post-interview state, never ends early during a rollout
        sim = PatientAgent(scenario=patient.scenario, model=patient.model, min_turns=10 ** 6)
        sim.set_history(snap)

        cum = 0.0
        rounds = []
        for r in range(1, num_rounds + 1):
            hist = format_dialogue(sim.get_history())
            best_msg, best_s = "", -1.0

            # Eq. 3: responses below tau_resp are discarded and regenerated, keep the best one
            for retry in range(max_retries + 1):
                temp = config.TEMP_SINGLE if retry == 0 else min(1.0, config.TEMP_SINGLE + 0.2)
                resp = therapist.respond_step(r, total_turns, temperature=temp)
                sc = evaluator.score_rollout(hist, resp, th)
                s = we_m * sc.get("empathy", 3.0) + wt_m * sc.get("alignment", 3.0)
                if s > best_s:
                    best_s, best_msg = s, resp
                if s >= tau_resp:
                    break

            cum += best_s
            therapist.add_message("assistant", best_msg)
            reaction = sim.respond(best_msg).replace("[SESSION_END]", "").strip()
            therapist.add_message("user", reaction)
            rounds.append({"round": r, "s_resp": round(best_s, 3)})

        return th, cum, rounds, sl

    # each therapy's rollout is independent, run them side by side
    with ThreadPoolExecutor(max_workers=max(1, min(config.PARALLEL_CALLS, len(selected_therapies)))) as ex:
        outs = list(ex.map(rollout, selected_therapies))

    # paper: the trajectory with the highest cumulative score is selected
    scores, logs = {}, {}
    for th, cum, rounds, sl in outs:
        scores[th] = cum
        logs[th] = {"score": round(cum, 3), "rounds": rounds, "slice": sl}
        print(f"  [stage3a] {th}: cumulative S_resp = {cum:.2f} ({[x['s_resp'] for x in rounds]})")

    # exact ties go to the stage 2 suitability score (the 14b scorer often gives every reply the same score)
    s_th = {c.get("therapy_type"): c.get("s_th", 0.0) for c in selected_therapies}
    win = max(scores, key=lambda k: (scores[k], s_th.get(k, 0.0))) if scores else selected_therapies[0]["therapy_type"]

    win_cand = next((c for c in selected_therapies if c.get("therapy_type") == win), selected_therapies[0])
    win_slice = lookup.get(win.lower()) or fallback_slice(win_cand)
    return win_slice, logs, scores.get(win, 0.0)

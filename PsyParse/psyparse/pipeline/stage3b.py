import copy
import json
from pathlib import Path

from ..agents.evaluation_agent import EvaluationAgent
from ..agents.patient_agent import PatientAgent
from ..agents.therapist_agent import TherapistAgent

# SOP Phase 5 / SOP Appendix B
_N_CANDIDATES = 4      # paper Section 4
_MAX_TURNS = 10        # our choice - paper doesn't specify
_WES = 0.5             # empathy weight (Eq. 4)
_WS = 0.5              # stability weight (Eq. 4)

_LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_TRANSCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "transcripts"


def _ssingle(scores):
    # Eq. 4: Ssingle = we,s * E + ws * S
    e = scores.get("empathy", 0)
    s = scores.get("stability", 0)
    return _WES * e + _WS * s


def run_stage3b(scenario, stage1_out, stage3a_out, stage2_out):
    """
    inputs:
      scenario      - raw scenario dict (for seeding both patient agents)
      stage1_out    - dict from run_stage1 (has 'profile', 'interview_history')
      stage3a_out   - dict from run_stage3a (has 'best_therapy')
      stage2_out    - dict from run_stage2 (has 'frameworks_list')

    returns dict with:
      transcript    - full list of {turn, therapist, patient} dicts
      turn_logs     - per-turn per-candidate scores
    """
    best_therapy_name = stage3a_out.get("best_therapy")
    frameworks = stage2_out.get("frameworks_list", [])
    post_interview_history = stage1_out.get("interview_history", [])
    profile = stage1_out.get("profile", {})
    profile_str = json.dumps(profile)

    # find the winning therapy slice from stage 2 frameworks
    best_fw = None
    for fw in frameworks:
        if fw.get("therapy") == best_therapy_name:
            best_fw = fw
            break

    if best_fw is None:
        print(f"[warn] stage3b: could not find framework for '{best_therapy_name}', using first available")
        best_fw = frameworks[0] if frameworks else {}

    # two separate patient agents - critical: different history objects
    # real patient: participates in the actual conversation
    real_patient = PatientAgent(scenario)
    real_patient.history = copy.deepcopy(post_interview_history)

    # simulated patient: only used for lookahead scoring, never touches real history
    sim_patient = PatientAgent(scenario)
    sim_patient.history = copy.deepcopy(post_interview_history)

    therapist = TherapistAgent()
    evaluator = EvaluationAgent()

    transcript = []
    turn_logs = []

    # opening patient message to start the real pruning session
    real_patient_msg = real_patient.send("Let's continue our session.", add_reminder=False)
    sim_patient.history = copy.deepcopy(real_patient.history)  # keep sim in sync

    for turn in range(_MAX_TURNS):
        # step 1: generate n=4 diverse candidates (stateless, doesn't touch therapist history)
        candidates = therapist.generate_candidates(real_patient_msg, best_fw, profile_str, n=_N_CANDIDATES)

        # step 2: for each candidate, get a simulated patient reaction
        sim_reactions = []
        sim_snap = sim_patient.snapshot()  # save sim state before lookahead
        for cand in candidates:
            sim_patient.restore(sim_snap)  # reset sim before each lookahead
            reaction = sim_patient.send(cand)
            sim_reactions.append(reaction)

        # restore sim to pre-lookahead state after all candidates scored
        sim_patient.restore(sim_snap)

        # step 3: batch score all 4 (Ri, Pi) pairs using Eq. 4
        try:
            batch_scores = evaluator.score_pruning_batch(
                therapist.history, candidates, sim_reactions
            )
        except Exception as e:
            print(f"[warn] turn {turn+1} eval failed ({e}), defaulting to equal scores")
            # default to equal scores so pipeline doesn't stall
            batch_scores = [{"empathy": 3, "stability": 3}] * _N_CANDIDATES

        # step 4: select R* = argmax Ssingle
        scored_pairs = [(i, _ssingle(batch_scores[i])) for i in range(_N_CANDIDATES)]
        best_idx, best_score = max(scored_pairs, key=lambda x: x[1])
        best_resp = candidates[best_idx]

        turn_log = {
            "turn": turn + 1,
            "patient_msg": real_patient_msg,
            "candidates": [
                {
                    "idx": i,
                    "response": candidates[i],
                    "sim_reaction": sim_reactions[i],
                    "scores": batch_scores[i],
                    "ssingle": round(_ssingle(batch_scores[i]), 4),
                }
                for i in range(_N_CANDIDATES)
            ],
            "selected_idx": best_idx,
            "selected_ssingle": round(best_score, 4),
        }
        turn_logs.append(turn_log)

        if best_score < 2.5:  # arbitrary low threshold - log it
            print(f"[warn] turn {turn+1}: all candidates scored low (best={best_score:.2f}), using best anyway")

        # step 5: commit R* to real therapist history, get real patient response
        therapist.history.append({"role": "user", "content": real_patient_msg})
        therapist.history.append({"role": "assistant", "content": best_resp})

        real_patient_reply = real_patient.send(best_resp)

        # keep sim in sync with real conversation after selection
        sim_patient.history = copy.deepcopy(real_patient.history)

        transcript.append({
            "turn": turn + 1,
            "therapist": best_resp,
            "patient": real_patient_reply,
        })

        # step 7: check for closure
        if "[SESSION_END]" in real_patient_reply:
            print(f"[info] stage3b: patient signalled SESSION_END at turn {turn+1}")
            break

        real_patient_msg = real_patient_reply

    print(f"[stage3b] pruning session complete: {len(transcript)} turns")

    # save logs and transcript
    _LOGS_DIR.mkdir(exist_ok=True)
    _TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    sid = scenario.get("dialog_id", "unknown")

    with open(_LOGS_DIR / f"pruning_scores_{sid}.json", "w") as f:
        json.dump(turn_logs, f, indent=2)

    with open(_TRANSCRIPTS_DIR / f"psyparse_{sid}.json", "w") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    return {
        "transcript": transcript,
        "turn_logs": turn_logs,
    }

import copy
import json
from pathlib import Path

from ..agents.evaluation_agent import EvaluationAgent
from ..agents.patient_agent import PatientAgent
from ..agents.therapist_agent import TherapistAgent

# SOP Phase 4 / SOP Table 7
_NUM_ROUNDS = 3
_MAX_RETRIES = 2       # SOP Step 3a.2(r3)
_TAU_RESP = 5.0        # discard threshold out of 10 (empathy + alignment, each 1-5, sum=10 max)
_WE = 0.5              # empathy weight (Eq. 3)
_WT = 0.5              # alignment weight (Eq. 3)

_LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


def _sresp(scores):
    # Eq. 3: Sresp = we,m * E + wt,m * A  (each on 1-5 scale)
    e = scores.get("empathy", 0)
    a = scores.get("alignment", 0)
    return _WE * e + _WT * a


def _run_candidate(therapy_slice, scenario, post_interview_history, profile_str):
    """run one therapy candidate for num_rounds, return (straj, round_log, conversation)."""
    therapist = TherapistAgent()
    patient = PatientAgent(scenario)

    # seed patient history with the post-interview exchange so it remembers context
    patient.history = copy.deepcopy(post_interview_history)

    evaluator = EvaluationAgent()

    straj = 0.0
    round_log = []
    therapy_name = therapy_slice.get("therapy", "unknown")

    # kick off with a short opening from patient to give therapist something to respond to
    patient_msg = patient.send("We can now begin the therapy session.", add_reminder=False)

    for r in range(_NUM_ROUNDS):
        best_resp = None
        best_score = -1.0
        accepted_scores = None

        for attempt in range(_MAX_RETRIES + 1):
            resp = therapist.send_rollout(patient_msg, therapy_slice, profile_str)
            raw_scores = evaluator.score_rollout(
                therapist.history, resp, therapy_name
            )
            sr = _sresp(raw_scores)

            if sr > best_score:
                best_score = sr
                best_resp = resp
                accepted_scores = raw_scores

            # accept if above threshold or we're on the last retry
            if sr >= _TAU_RESP or attempt == _MAX_RETRIES:
                if attempt > 0 and sr < _TAU_RESP:
                    print(
                        f"[warn] {therapy_name} round {r+1}: "
                        f"all retries below tau_resp ({best_score:.2f}<{_TAU_RESP}), "
                        f"using best attempt"
                    )
                break

        # commit best response to therapist history
        therapist.history.append({"role": "user", "content": patient_msg})
        therapist.history.append({"role": "assistant", "content": best_resp})

        straj += best_score
        round_log.append({
            "round": r + 1,
            "therapist_response": best_resp,
            "scores": accepted_scores,
            "sresp": round(best_score, 4),
        })

        # patient reacts to accepted response - check for session end
        patient_msg = patient.send(best_resp)
        if "[SESSION_END]" in patient_msg:
            print(f"[info] {therapy_name}: patient signalled SESSION_END at round {r+1}")
            break

    return straj, round_log, therapist.history


def run_stage3a(scenario, stage1_out, stage2_out):
    """
    inputs:
      scenario        - raw scenario dict (for re-seeding patient)
      stage1_out      - dict from run_stage1 (has 'profile', 'history')
      stage2_out      - dict from run_stage2 (has 'frameworks_list', 'top_k2')

    returns dict with:
      best_therapy    - therapy name string
      best_straj      - float score
      rollout_log     - per-candidate per-round scores
    """
    frameworks = stage2_out.get("frameworks_list", [])
    post_interview_history = stage1_out.get("interview_history", [])
    profile = stage1_out.get("profile", {})
    profile_str = json.dumps(profile)

    if not frameworks:
        print("[warn] stage3a: no frameworks from stage2, skipping rollout")
        return {"best_therapy": None, "best_straj": 0.0, "rollout_log": []}

    rollout_log = []
    best_therapy = None
    best_straj = -1.0

    for fw in frameworks:
        therapy_name = fw.get("therapy", "unknown")
        print(f"[stage3a] rolling out therapy: {therapy_name}")

        try:
            straj, round_log, _ = _run_candidate(
                fw, scenario, post_interview_history, profile_str
            )
        except Exception as e:
            print(f"[error] {therapy_name} rollout failed: {e}")
            straj = 0.0
            round_log = []

        print(f"  Straj({therapy_name}) = {straj:.4f}")
        rollout_log.append({
            "therapy": therapy_name,
            "straj": round(straj, 4),
            "rounds": round_log,
        })

        if straj > best_straj:
            best_straj = straj
            best_therapy = therapy_name

    print(f"[stage3a] T* = '{best_therapy}' (Straj={best_straj:.4f})")

    # save logs
    _LOGS_DIR.mkdir(exist_ok=True)
    sid = scenario.get("dialog_id", "unknown")
    log_path = _LOGS_DIR / f"rollout_scores_{sid}.json"
    with open(log_path, "w") as f:
        json.dump(rollout_log, f, indent=2)

    return {
        "best_therapy": best_therapy,
        "best_straj": best_straj,
        "rollout_log": rollout_log,
    }

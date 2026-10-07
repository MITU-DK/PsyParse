import json
import sys
from pathlib import Path

# project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psyparse.agents.evaluation_agent import _parse_json, _strip_md
from psyparse.agents.patient_agent import PatientAgent
from psyparse.agents.therapist_agent import TherapistAgent
from psyparse.agents.base_agent import BaseAgent
from psyparse.pipeline.stage1 import run_stage1
from psyparse.pipeline.stage2 import run_stage2
from psyparse.pipeline.stage3a import run_stage3a
from psyparse.pipeline.stage3b import run_stage3b

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EVAL_DIR = Path(__file__).resolve().parent
TRANSCRIPTS_DIR = Path(__file__).resolve().parent.parent / "transcripts"

_MAX_TURNS_BASELINE = 10
_METRICS = ["coherence", "completeness", "humaneness", "technique", "structure", "context_retention"]

# load judge prompt once
_JUDGE_PROMPT = (EVAL_DIR / "judge_prompt.txt").read_text()


def _judge_transcript(transcript, eval_agent):
    # transcript: list of {turn, therapist, patient}
    convo_text = "\n".join(
        f"Turn {t['turn']}\nTherapist: {t['therapist']}\nPatient: {t['patient']}"
        for t in transcript
    )
    msgs = [
        {"role": "system", "content": _JUDGE_PROMPT},
        {"role": "user", "content": f"Conversation transcript:\n{convo_text}"},
    ]
    raw = eval_agent.generate_from(msgs, temp=0.0)
    return _parse_json(raw)


def _run_baseline(scenario):
    """standard counselor, no RAG, no rollout, no pruning - max 10 turns."""
    topic = scenario.get("topic", "")
    background = scenario.get("background", "")

    therapist = TherapistAgent()
    therapist.setup_baseline(topic, background)

    patient = PatientAgent(scenario)

    transcript = []
    patient_msg = patient.send("Hello, I'd like to start our session.", add_reminder=False)

    for turn in range(_MAX_TURNS_BASELINE):
        therapist_reply = therapist.send(patient_msg)
        patient_reply = patient.send(therapist_reply)

        transcript.append({
            "turn": turn + 1,
            "therapist": therapist_reply,
            "patient": patient_reply,
        })

        if "[SESSION_END]" in patient_reply:
            print(f"  [baseline] patient SESSION_END at turn {turn+1}")
            break

        patient_msg = patient_reply

    return transcript


def _run_psyparse(scenario):
    """full psyparse pipeline - returns stage3b transcript."""
    s1 = run_stage1(scenario)
    s2 = run_stage2(
        embed_profile=s1["embed_profile"],
        keywords=s1["keywords"],
        full_profile=s1["profile"],
    )
    s3a = run_stage3a(scenario, s1, s2)
    s3b = run_stage3b(scenario, s1, s3a, s2)
    return s3b["transcript"], s1, s2, s3a


def run_eval(n_scenarios=None):
    """
    main A/B evaluation loop.
    n_scenarios: int or None (None = run all in eval_scenarios.json)
    """
    eval_path = DATA_DIR / "eval_scenarios.json"
    if not eval_path.exists():
        sys.exit(f"[error] {eval_path} not found")

    with open(eval_path, encoding="utf-8") as f:
        scenarios = json.load(f)

    if n_scenarios:
        scenarios = scenarios[:n_scenarios]

    print(f"running eval on {len(scenarios)} scenarios")

    # one shared eval agent for judging - deterministic, temp=0
    from psyparse.agents.evaluation_agent import EvaluationAgent
    judge = EvaluationAgent()

    RESULTS_DIR.mkdir(exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)

    all_results = []

    for idx, scenario in enumerate(scenarios):
        sid = scenario.get("dialog_id", f"eval_{idx}")
        topic = scenario.get("topic", "unknown")
        print(f"\n--- eval {idx+1}/{len(scenarios)}: {sid} | {topic} ---")

        result = {"scenario_id": sid, "topic": topic}

        # A: baseline
        print("  [A] running baseline...")
        try:
            baseline_transcript = _run_baseline(scenario)
            baseline_scores = _judge_transcript(baseline_transcript, judge)
            result["baseline_scores"] = baseline_scores
            result["baseline_turns"] = len(baseline_transcript)

            # save baseline transcript
            with open(TRANSCRIPTS_DIR / f"baseline_{sid}.json", "w", encoding="utf-8") as f:
                json.dump(baseline_transcript, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [error] baseline failed: {e}")
            result["baseline_scores"] = None

        # B: psyparse full pipeline
        print("  [B] running psyparse...")
        try:
            psyparse_transcript, s1, s2, s3a = _run_psyparse(scenario)
            psyparse_scores = _judge_transcript(psyparse_transcript, judge)
            result["psyparse_scores"] = psyparse_scores
            result["psyparse_turns"] = len(psyparse_transcript)
            result["best_therapy"] = s3a.get("best_therapy")
            result["top_k2"] = [c["therapy_type"] for c in s2.get("top_k2", [])]
        except Exception as e:
            print(f"  [error] psyparse failed: {e}")
            result["psyparse_scores"] = None

        # delta: psyparse - baseline
        if result.get("baseline_scores") and result.get("psyparse_scores"):
            delta = {}
            for m in _METRICS:
                b = result["baseline_scores"].get(m, 0)
                p = result["psyparse_scores"].get(m, 0)
                delta[m] = round(p - b, 2)
            result["delta"] = delta
            print(f"  delta: {delta}")
        else:
            result["delta"] = None

        all_results.append(result)

        # save incrementally so partial runs aren't lost
        with open(RESULTS_DIR / "eval_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\ndone. results -> {RESULTS_DIR / 'eval_results.json'}")
    return all_results


if __name__ == "__main__":
    run_eval()

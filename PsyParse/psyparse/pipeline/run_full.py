import json
import logging
from typing import Dict, Any, Optional
import config
from psyparse.pipeline.stage1 import run_stage_1
from psyparse.pipeline.stage2 import run_stage_2
from psyparse.pipeline.stage3a import run_stage_3a
from psyparse.pipeline.stage3b import run_stage_3b
from psyparse.retrieval.hybrid_search import HybridRetriever
from psyparse.agents.evaluation_agent import EvaluationAgent
from psyparse.agents.therapist_agent import TherapistAgent, support_for

logger = logging.getLogger(__name__)

def run_psyparse_pipeline(
    scenario: Dict[str, Any],
    retriever: HybridRetriever,
    model: Optional[str] = None,
    max_therapy_turns: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes the end-to-end PsyPARSE counseling pipeline.
    """
    model = model or config.DEEPSEEK_MODEL
    turns = max_therapy_turns or config.THERAPY_TURNS
    evaluator = EvaluationAgent(model=model)

    print("\n--- Starting PsyParse Pipeline ---")
    print("Running Stage 1: Patient Interview & Profile Extraction...")
    profile, keywords, intake_history, patient_agent = run_stage_1(
        scenario=scenario,
        patient_model=model,
        therapist_model=model,
        retriever=retriever,
    )
    print(f"Stage 1 Complete. keywords={keywords}")

    print("Running Stage 2: Multi-Therapy RAG & Guidance Synthesis...")
    # not the interview agent: its "speak to the client, no lists" prompt fights the json output
    therapist_agent = TherapistAgent(mode="guided", model=model)
    selected_therapies, guidance = run_stage_2(
        patient_profile=profile,
        keywords=keywords,
        retriever=retriever,
        therapist=therapist_agent,
    )
    print(f"Stage 2 Complete. top therapies={[(t['therapy_type'], round(t['s_th'], 3)) for t in selected_therapies]}")

    print(f"Running Stage 3a: Multi-Turn Rollout & Selection (Trying {len(selected_therapies)} candidates)...")
    best_slice, rollout_logs, traj_score = run_stage_3a(
        selected_therapies=selected_therapies,
        guidance_framework=guidance,
        patient_profile=profile,
        patient=patient_agent,
        evaluator=evaluator,
        total_turns=turns,
    )
    print(f"Stage 3a Complete. T* = {best_slice.get('therapy')}")

    print(f"Running Stage 3b: Pruned Therapy Dialogue (Using {best_slice.get('therapy', 'selected therapy')})...")
    support = support_for(guidance, best_slice.get("therapy", ""))
    full_transcript, pruning_logs = run_stage_3b(
        best_therapy_slice=best_slice,
        patient_profile=profile,
        real_patient=patient_agent,
        evaluator=evaluator,
        max_turns=turns,
        min_turns=min(turns, config.MIN_THERAPY_TURNS),
        support=support,
    )
    print("Stage 3b Complete. Pipeline Finished!\n")

    return {
        "scenario_id": scenario.get("dialog_id", "unknown"),
        "topic": scenario.get("topic", "General"),
        "patient_profile": profile,
        "keywords": keywords,
        "top_therapies": [
            {k: t.get(k) for k in ["therapy_type", "case_id", "topic", "score", "m_s", "app", "s_th"]}
            for t in selected_therapies
        ],
        "guidance": guidance,
        "best_therapy": best_slice.get("therapy", "CBT"),
        "best_slice": best_slice,
        "trajectory_score": traj_score,
        "rollout_logs": rollout_logs,
        "pruning_logs": pruning_logs,
        "intake_turns": len([m for m in intake_history if m["role"] == "user"]),
        "transcript": full_transcript,
        "turns": len([m for m in full_transcript if m["role"] == "user"]),
    }


if __name__ == "__main__":
    # quick demo on the first dev scenario
    with open(config.DATA_DIR / "dev_scenarios.json", encoding="utf-8") as f:
        sc = json.load(f)[0]
    res = run_psyparse_pipeline(sc, HybridRetriever())
    from psyparse.agents.base_agent import format_dialogue
    print(format_dialogue(res["transcript"]))
    out = config.ROOT / "test_outputs" / f"{res['scenario_id']}_result.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("saved ->", out)

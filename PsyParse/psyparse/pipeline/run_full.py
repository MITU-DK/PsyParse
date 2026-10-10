import os
import json
import logging
from typing import Dict, Any, Tuple
from psyparse.pipeline.stage1 import run_stage_1
from psyparse.pipeline.stage2 import run_stage_2
from psyparse.pipeline.stage3a import run_stage_3a
from psyparse.pipeline.stage3b import run_stage_3b
from psyparse.retrieval.hybrid_search import HybridRetriever
from psyparse.agents.evaluation_agent import EvaluationAgent
from psyparse.agents.therapist_agent import TherapistAgent

logger = logging.getLogger(__name__)

def run_psyparse_pipeline(
    scenario: Dict[str, Any],
    retriever: HybridRetriever,
    model: str = "qwen2.5:14b",
    max_therapy_turns: int = 5,
) -> Dict[str, Any]:
    """
    Executes the end-to-end PsyPARSE counseling pipeline.
    """
    evaluator = EvaluationAgent(model=model)

    # Stage 1: Assessment Interview & Profile Extraction
    print("\n--- Starting PsyParse Pipeline ---")
    print("Running Stage 1: Patient Interview & Profile Extraction...")
    profile, keywords, intake_history, patient_agent = run_stage_1(
        scenario=scenario,
        patient_model=model,
        therapist_model=model,
    )
    print("Stage 1 Complete.")

    # Stage 2: Multi-Therapy RAG & Framework Synthesis
    print("Running Stage 2: Multi-Therapy RAG & Guidance Synthesis...")
    therapist_agent = TherapistAgent(mode="interview", model=model)
    selected_therapies, guidance = run_stage_2(
        patient_profile=profile,
        keywords=keywords,
        retriever=retriever,
        therapist=therapist_agent,
    )
    print("Stage 2 Complete.")

    # Stage 3a: Multi-Turn Rollout & Selection of T*
    print(f"Running Stage 3a: Multi-Turn Rollout & Selection (Trying {len(selected_therapies)} candidates)...")
    best_slice, rollout_logs, traj_score = run_stage_3a(
        selected_therapies=selected_therapies,
        guidance_framework=guidance,
        patient_profile=profile,
        patient=patient_agent,
        evaluator=evaluator,
    )
    print("Stage 3a Complete.")

    # Stage 3b: Pruned Therapy Dialogue
    print(f"Running Stage 3b: Pruned Therapy Dialogue (Using {best_slice.get('therapy', 'selected therapy')})...")
    full_transcript = run_stage_3b(
        best_therapy_slice=best_slice,
        patient_profile=profile,
        real_patient=patient_agent,
        evaluator=evaluator,
        max_turns=max_therapy_turns,
        min_turns=3,
    )
    print("Stage 3b Complete. Pipeline Finished!\n")

    return {
        "scenario_id": scenario.get("dialog_id", "unknown"),
        "topic": scenario.get("topic", "General"),
        "patient_profile": profile,
        "keywords": keywords,
        "top_therapies": [t["therapy_type"] for t in selected_therapies],
        "best_therapy": best_slice.get("therapy", "CBT"),
        "trajectory_score": traj_score,
        "transcript": full_transcript,
        "turns": len([m for m in full_transcript if m["role"] == "user"])
    }
import logging
from typing import Dict, Any, List, Tuple
from psyparse.retrieval.hybrid_search import HybridRetriever
from psyparse.agents.therapist_agent import TherapistAgent

logger = logging.getLogger(__name__)

def run_stage_2(
    patient_profile: Dict[str, Any],
    keywords: List[str],
    retriever: HybridRetriever,
    therapist: TherapistAgent,
    alpha: float = 0.5,
    w1: float = 0.5,
    w2: float = 0.5,
    k1: int = 10,
    k2: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes Stage 2: Hybrid RAG, therapy scoring (Eq. 1 & 2),
    percentile filtering, and multi-therapy guidance synthesis.
    """
    selected_therapies = retriever.retrieve(
        patient_profile=patient_profile,
        keywords=keywords,
        alpha=alpha,
        w1=w1,
        w2=w2,
        k1=k1,
        k2=k2,
    )
    guidance_framework = therapist.synthesize_guidance(selected_therapies, patient_profile)
    return selected_therapies, guidance_framework
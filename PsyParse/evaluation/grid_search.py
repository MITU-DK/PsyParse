import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from psyparse.retrieval.hybrid_search import HybridRetriever
from psyparse.pipeline.stage1 import run_stage_1

logger = logging.getLogger(__name__)

def run_retrieval_grid_search(
    dev_scenarios_path: str = "data/dev_scenarios.json",
    retriever_instance: Optional[HybridRetriever] = None,
) -> Dict[str, float]:
    """
    Empirically evaluates alpha balance parameters {0.3, 0.5, 0.7}
    on Dev scenarios to avoid saturated 100.0 outputs.
    """
    with open(dev_scenarios_path, "r", encoding="utf-8") as f:
        dev_scenarios = json.load(f)[:3]

    retriever = retriever_instance or HybridRetriever()
    alpha_grid = [0.3, 0.5, 0.7]
    results: Dict[str, float] = {}

    # Pre-compute Stage 1 profiles to save API calls
    precomputed = []
    print("Pre-computing Phase 1 interviews (doing this once per scenario)...")
    for idx, sc in enumerate(dev_scenarios):
        print(f"Running interview for scenario {idx + 1}/{len(dev_scenarios)}...")
        profile, keywords, _, _ = run_stage_1(sc)
        precomputed.append((sc, profile, keywords))

    print("Executing Alpha Grid Search...")
    for alpha in alpha_grid:
        topic_match_scores = []
        for idx, (sc, profile, keywords) in enumerate(precomputed):
            print(f"Testing alpha={alpha}, scenario {idx + 1}/{len(dev_scenarios)}...")
            retrieved = retriever.retrieve(profile, keywords, alpha=alpha, k1=10, k2=3)

            target_topic = sc.get("topic", "").lower()
            matches = sum(
                1 for r in retrieved
                if any(target_topic in cond.lower() for cond in r.get("applicable_conditions", []))
            )
            topic_match_scores.append(matches / max(1, len(retrieved)))

        score = float(np.mean(topic_match_scores) * 100.0)
        results[f"alpha_{alpha:.1f}"] = round(score, 2)

    with open("grid_search_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    print(run_retrieval_grid_search())
import json
import itertools
import logging
from typing import Dict, Any, Optional

import numpy as np

import config
from psyparse.retrieval.hybrid_search import HybridRetriever
from psyparse.pipeline.stage1 import run_stage_1

logger = logging.getLogger(__name__)

# dataset label -> therapy_type used in the db (same mapping as data prep)
GOLD_MAP = {
    "solution-focused brief therapy": "SFBT",
    "psychoanalytic therapy": "Psychoanalytic",
    "cognitive behaviour therapy": "CBT",
    "cognitive behavioral therapy": "CBT",
    "client-centered therapy": "PCT",
    "adlerian therapy": "Adlerian",
    "family therapy": "Family",
    "acceptance commitment therapy": "ACT",
    "marriage and family systems therapy": "Family Systems",
    "rational emotive behavior therapy": "REBT",
    "gestalt therapy": "Gestalt",
    "reality therapy": "Reality",
    "postmodern therapy": "Postmodern",
    "existential therapy": "Existential",
}


def run_retrieval_grid_search(retriever: Optional[HybridRetriever] = None) -> Dict[str, Any]:
    """
    Grid search over alpha and (w1, w2) on the dev scenarios (SOP Table VII).
    Objective: is the dev scenario's real therapy in the top-k2 retrieved types (hit),
    tie-break by reciprocal rank of it.
    """
    with open(config.DATA_DIR / "dev_scenarios.json", encoding="utf-8") as f:
        dev = json.load(f)

    retriever = retriever or HybridRetriever()

    # stage 1 is the only llm part, cache it so reruns are free
    cache_path = config.RESULTS_DIR / "dev_profiles.json"
    cache = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    for idx, sc in enumerate(dev):
        sid = sc.get("dialog_id", str(idx))
        if sid in cache:
            continue
        print(f"Running interview for dev scenario {idx + 1}/{len(dev)}...")
        profile, keywords, _, _ = run_stage_1(sc, retriever=retriever)
        cache[sid] = {"profile": profile, "keywords": keywords}
        config.RESULTS_DIR.mkdir(exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    alphas = [0.3, 0.5, 0.7]
    ws = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]
    rows = []
    for alpha, (w1, w2) in itertools.product(alphas, ws):
        hits, rr = [], []
        for idx, sc in enumerate(dev):
            sid = sc.get("dialog_id", str(idx))
            gold = GOLD_MAP.get(str(sc.get("psychotherapy", "")).lower().strip(), sc.get("psychotherapy"))
            got = retriever.retrieve(cache[sid]["profile"], cache[sid]["keywords"], alpha=alpha, w1=w1, w2=w2)
            types = [g["therapy_type"] for g in got]
            hits.append(1.0 if gold in types else 0.0)
            rr.append(1.0 / (types.index(gold) + 1) if gold in types else 0.0)
        rows.append({"alpha": alpha, "w1": w1, "w2": w2,
                     "hit_at_k2": round(float(np.mean(hits)) * 100, 2), "mrr": round(float(np.mean(rr)), 3)})
        print(f"alpha={alpha} w1={w1} w2={w2} -> hit@k2={rows[-1]['hit_at_k2']}% mrr={rows[-1]['mrr']}")

    # prefer the paper/SOP default 0.5/0.5/0.5 when scores tie
    def key(r):
        return (r["hit_at_k2"], r["mrr"], -abs(r["alpha"] - 0.5) - abs(r["w1"] - 0.5))
    best = max(rows, key=key)
    out = {"n_dev": len(dev), "grid": rows, "best": best}
    with open(config.RESULTS_DIR / "grid_search_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nbest: {best}\n-> set ALPHA, W1, W2 in config.py to these values")
    return out


if __name__ == "__main__":
    run_retrieval_grid_search()

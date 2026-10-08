import json
import numpy as np
from typing import List, Dict, Any

METRIC_MAP = {
    "coherence": "Coh.",
    "completeness": "Cpl.",
    "humaneness": "Hum.",
    "technique": "Tec.",
    "structure": "Str.",
    "context_retention": "Ctx.",
}

def aggregate_eval_results(results_path: str = "eval_results.json") -> str:
    with open(results_path, "r", encoding="utf-8") as f:
        results: List[Dict[str, Any]] = json.load(f)

    if not results:
        return "No evaluation results available."

    lines = [
        f"# PsyPARSE Evaluation Summary Across N={len(results)} Scenarios\n",
        "| Metric | Baseline Mean | PsyPARSE Mean | Mean Δ | StdDev Δ | Target Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    targets = {
        "coherence": 5.0,
        "completeness": 8.0,
        "humaneness": 5.0,
        "technique": 10.0,
        "structure": 8.0,
        "context_retention": 10.0,
    }

    for key, abbr in METRIC_MAP.items():
        base_vals = [r["baseline_scores"][key] for r in results]
        psy_vals = [r["psyparse_scores"][key] for r in results]
        deltas = [r["delta"][key] for r in results]

        b_mean = np.mean(base_vals)
        p_mean = np.mean(psy_vals)
        d_mean = np.mean(deltas)
        d_std = np.std(deltas)

        tgt = targets[key]
        status = "PASSED" if d_mean >= tgt else "BELOW TARGET"

        lines.append(f"| {abbr} | {b_mean:.2f} | {p_mean:.2f} | {d_mean:+.2f} | {d_std:.2f} | {status} |")

    table_md = "\n".join(lines)
    import os
    os.makedirs("results", exist_ok=True)
    with open("results/summary_table.md", "w", encoding="utf-8") as f:
        f.write(table_md)

    return table_md

if __name__ == "__main__":
    print(aggregate_eval_results())
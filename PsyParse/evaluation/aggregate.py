import json
import operator
from collections import Counter
from typing import List, Dict, Any

import numpy as np

import config

METRIC_MAP = {
    "coherence": "Coh.",
    "completeness": "Cpl.",
    "humaneness": "Hum.",
    "technique": "Tec.",
    "structure": "Str.",
    "context_retention": "Ctx.",
}

# SOP targets on the delta, with the exact operators from the SOP
TARGETS = {
    "coherence": (">", 5.0),
    "completeness": (">", 8.0),
    "humaneness": (">=", 5.0),
    "technique": (">", 10.0),
    "structure": (">=", 8.0),
    "context_retention": (">=", 10.0),
}
OPS = {">": operator.gt, ">=": operator.ge}


def aggregate_eval_results(results_path: str = str(config.RESULTS_DIR / "eval_results.json")) -> str:
    with open(results_path, "r", encoding="utf-8") as f:
        results: List[Dict[str, Any]] = json.load(f)

    valid = [r for r in results if r.get("valid", True) and r.get("delta")]
    if not valid:
        return "No valid evaluation results available."

    lines = [
        f"# PsyPARSE Evaluation Summary (valid scenarios: {len(valid)}/{len(results)})\n",
        "**QUICK TEST run (reduced settings) - only checks that the pipeline works, not real results.**\n" if any(r.get("quick_test") for r in valid) else "",
        f"agent model: `{valid[0].get('agent_model', '?')}` | judge model: `{valid[0].get('judge_model', '?')}`\n",
        "| Metric | Baseline Mean | PsyPARSE Mean | Mean Δ | StdDev Δ | SOP Target | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for key, abbr in METRIC_MAP.items():
        b_mean = np.mean([r["baseline_scores"][key] for r in valid])
        p_mean = np.mean([r["psyparse_scores"][key] for r in valid])
        deltas = [r["delta"][key] for r in valid]
        d_mean, d_std = np.mean(deltas), np.std(deltas)

        op, tgt = TARGETS[key]
        status = "PASSED" if OPS[op](d_mean, tgt) else "BELOW TARGET"
        lines.append(f"| {abbr} | {b_mean:.2f} | {p_mean:.2f} | {d_mean:+.2f} | {d_std:.2f} | {op} +{tgt:g} | {status} |")

    wins = sum(1 for r in valid if np.mean(list(r["delta"].values())) > 0)
    lines.append(f"\nPsyPARSE better on average in {wins}/{len(valid)} scenarios.")
    lines.append(f"Chosen therapies (T*): {dict(Counter(r.get('best_therapy') for r in valid))}")

    table_md = "\n".join(lines)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    with open(config.RESULTS_DIR / "summary_table.md", "w", encoding="utf-8") as f:
        f.write(table_md)

    return table_md


if __name__ == "__main__":
    print(aggregate_eval_results())

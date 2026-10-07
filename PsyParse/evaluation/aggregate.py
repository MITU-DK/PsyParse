import json
import statistics
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
_METRICS = ["coherence", "completeness", "humaneness", "technique", "structure", "context_retention"]
_METRIC_LABELS = {
    "coherence": "Coh.",
    "completeness": "Cpl.",
    "humaneness": "Hum.",
    "technique": "Tec.",
    "structure": "Str.",
    "context_retention": "Ctx.",
}


def aggregate(results_path=None):
    path = Path(results_path) if results_path else RESULTS_DIR / "eval_results.json"
    if not path.exists():
        sys.exit(f"[error] {path} not found - run run_eval.py first")

    with open(path, encoding="utf-8") as f:
        results = json.load(f)

    # collect deltas per metric, skip failed scenarios
    deltas = {m: [] for m in _METRICS}
    baseline_avgs = {m: [] for m in _METRICS}
    psyparse_avgs = {m: [] for m in _METRICS}
    valid_count = 0

    for r in results:
        if r.get("delta") is None:
            continue
        valid_count += 1
        for m in _METRICS:
            deltas[m].append(r["delta"].get(m, 0))
            if r.get("baseline_scores"):
                baseline_avgs[m].append(r["baseline_scores"].get(m, 0))
            if r.get("psyparse_scores"):
                psyparse_avgs[m].append(r["psyparse_scores"].get(m, 0))

    print(f"\nAggregated over {valid_count}/{len(results)} valid scenarios\n")

    # build summary table
    lines = []
    lines.append(f"{'Metric':<14} {'Baseline':>10} {'PsyPARSE':>10} {'Δ Mean':>10} {'Δ StdDev':>10}")
    lines.append("-" * 58)

    for m in _METRICS:
        label = _METRIC_LABELS[m]
        b_mean = statistics.mean(baseline_avgs[m]) if baseline_avgs[m] else 0
        p_mean = statistics.mean(psyparse_avgs[m]) if psyparse_avgs[m] else 0
        d_mean = statistics.mean(deltas[m]) if deltas[m] else 0
        d_std = statistics.stdev(deltas[m]) if len(deltas[m]) > 1 else 0
        lines.append(
            f"{label:<14} {b_mean:>10.2f} {p_mean:>10.2f} {d_mean:>+10.2f} {d_std:>10.2f}"
        )

    table = "\n".join(lines)
    print(table)

    # save to file
    summary_path = RESULTS_DIR / "summary_table.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# PsyPARSE Evaluation Summary\n\n")
        f.write(f"Valid scenarios: {valid_count}/{len(results)}\n\n")
        f.write("```\n")
        f.write(table)
        f.write("\n```\n")

    print(f"\nsummary saved -> {summary_path}")

    # also print per-scenario delta summary
    print("\nPer-scenario best_therapy selections:")
    for r in results:
        sid = r.get("scenario_id", "?")
        therapy = r.get("best_therapy", "N/A")
        status = "ok" if r.get("delta") else "FAILED"
        print(f"  {sid}: {therapy} [{status}]")

    return deltas


if __name__ == "__main__":
    aggregate()

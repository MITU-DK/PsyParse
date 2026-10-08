import json
from pathlib import Path
from psyparse.pipeline import stage3a, stage3b
from evaluation.run_eval import _run_psyparse, _judge_transcript
from psyparse.agents.evaluation_agent import EvaluationAgent

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

def run_grid_search():
    with open(DATA_DIR / "dev_scenarios.json", "r", encoding="utf-8") as f:
        scenarios = json.load(f)[:3]  # use 3 dev scenarios

    # grid search: w_e (empathy) vs w_t (alignment/stability)
    weights = [
        (0.2, 0.8),
        (0.5, 0.5),
        (0.8, 0.2)
    ]
    
    results = {}
    
    for w_e, w_t in weights:
        print(f"\n--- Testing weights: w_e={w_e}, w_t={w_t} ---")
        
        # monkey-patch the weights
        stage3a._WE = w_e
        stage3a._WT = w_t
        stage3b._WES = w_e
        stage3b._WS = w_t
        
        scores = []
        for scenario in scenarios:
            print(f"  [scenario {scenario['dialog_id']}]")
            try:
                transcript, _, _, _ = _run_psyparse(scenario)
                # score the transcript
                judge = EvaluationAgent()
                judge_res = _judge_transcript(transcript, judge)
                # average across the 6 metrics
                avg_score = sum(judge_res.values()) / 6
                scores.append(avg_score)
            except Exception as e:
                print(f"  failed: {e}")
                
        avg_overall = sum(scores) / len(scores) if scores else 0
        results[f"{w_e}_{w_t}"] = avg_overall
        print(f"  -> Avg Score: {avg_overall:.2f}")

    print("\n--- Grid Search Results ---")
    for k, v in results.items():
        print(f"Weights {k}: {v:.2f}")
        
    with open(RESULTS_DIR / "grid_search_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_grid_search()

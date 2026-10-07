import json
import sys
from pathlib import Path

from .stage1 import run_stage1
from .stage2 import run_stage2

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "test_outputs"


def load_scenarios(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_result(idx, scenario, stage1_out, stage2_out):
    OUT_DIR.mkdir(exist_ok=True)
    result = {
        "scenario_id": scenario.get("dialog_id", f"scenario_{idx}"),
        "topic": scenario.get("topic", ""),
        "profile": stage1_out["profile"],
        "keywords": stage1_out["keywords"],
        "top_k2": stage2_out["top_k2"],
        "framework": stage2_out["framework"],
    }
    out_path = OUT_DIR / f"scenario_{idx}_{result['scenario_id']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  saved -> {out_path}")
    return result


def run_pipeline(scenario):
    stage1_out = run_stage1(scenario)
    stage2_out = run_stage2(
        embed_profile=stage1_out["embed_profile"],
        keywords=stage1_out["keywords"],
        full_profile=stage1_out["profile"],
    )
    return stage1_out, stage2_out


def main():
    # --skip-stage3 flag: run only Stage 1+2 (fast dev mode, fewer tokens)
    skip_stage3 = "--skip-stage3" in sys.argv

    # use dev_scenarios.json - NEVER eval_scenarios.json until phase 6
    scenarios_path = DATA_DIR / "dev_scenarios.json"
    if not scenarios_path.exists():
        sys.exit(f"[error] {scenarios_path} not found - run data_prep.py first")

    scenarios = load_scenarios(scenarios_path)
    print(f"loaded {len(scenarios)} dev scenarios")
    if skip_stage3:
        print("[info] --skip-stage3 active: running Stage 1+2 only")

    for idx, scenario in enumerate(scenarios):
        topic = scenario.get("topic", "unknown")
        sid = scenario.get("dialog_id", f"scenario_{idx}")
        print(f"\n--- scenario {idx+1}/{len(scenarios)}: {sid} | topic: {topic} ---")

        try:
            stage1_out, stage2_out = run_pipeline(scenario)
            result = save_result(idx + 1, scenario, stage1_out, stage2_out)

            # quick inspection output for manual review
            print(f"  profile fields: {list(result['profile'].keys())}")
            print(f"  keywords: {result['keywords']}")
            therapies = [c['therapy_type'] for c in result['top_k2']]
            print(f"  top therapies: {therapies}")
            fw_types = [f.get('therapy') for f in result['framework'].get('frameworks', [])]
            print(f"  framework types: {fw_types}")

            if skip_stage3:
                continue

            # stage 3a: multi-turn rollout -> select best therapy T*
            from .stage3a import run_stage3a
            stage3a_out = run_stage3a(scenario, stage1_out, stage2_out)
            print(f"  best therapy (T*): {stage3a_out['best_therapy']} (Straj={stage3a_out['best_straj']:.4f})")

            # stage 3b: response pruning -> final conversation
            from .stage3b import run_stage3b
            stage3b_out = run_stage3b(scenario, stage1_out, stage3a_out, stage2_out)
            print(f"  pruning turns completed: {len(stage3b_out['transcript'])}")

            # append stage3 results to saved file
            result["best_therapy"] = stage3a_out["best_therapy"]
            result["best_straj"] = stage3a_out["best_straj"]
            result["pruning_turns"] = len(stage3b_out["transcript"])
            out_path = OUT_DIR / f"scenario_{idx+1}_{sid}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"  [error] scenario {sid} failed: {e}")
            continue

    print(f"\ndone. outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()

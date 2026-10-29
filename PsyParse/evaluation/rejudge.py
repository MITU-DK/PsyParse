import sys
import json
import glob
import re
from pathlib import Path

import config
from evaluation.run_eval import METRICS, make_judge, judge_transcript

# re-score saved transcripts with the judge set in .env (JUDGE_MODEL etc), no pipeline run needed
# usage: PYTHONPATH=. python evaluation/rejudge.py [transcripts_dir] [--resume]


def to_text(msgs):
    # same string the judge saw in run_eval (format_dialogue on the clean history)
    return "\n".join(f"{m['speaker']}: {m['text']}" for m in msgs)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tdir = Path(args[0]) if args else config.RESULTS_DIR / "transcripts"
    files = sorted(glob.glob(str(tdir / "*.json")))
    if not files:
        print(f"no transcripts in {tdir}")
        return

    name = re.sub(r"[^a-zA-Z0-9.]+", "_", config.JUDGE_MODEL)
    out = config.RESULTS_DIR / f"rejudge_{name}.json"
    res = []
    if "--resume" in sys.argv and out.exists():
        res = json.load(open(out))
        print(f"resuming, {len(res)} already judged")
    done = {r["scenario_id"] for r in res}

    print(f"judge = {config.JUDGE_MODEL} | samples = {config.JUDGE_SAMPLES} | {len(files)} transcripts from {tdir}")
    judge = make_judge()
    for f in files:
        d = json.load(open(f))
        sid = d["scenario_id"]
        if sid in done:
            continue
        print(f"\n{sid} ({d.get('topic', '')})")
        b = judge_transcript(judge, to_text(d["baseline_transcript"]))
        p = judge_transcript(judge, to_text(d["psyparse_transcript"]))
        ok = b is not None and p is not None
        res.append({
            "scenario_id": sid,
            "best_therapy": d.get("best_therapy"),
            "valid": ok,
            "baseline_scores": b,
            "psyparse_scores": p,
            "delta": {k: round(p[k] - b[k], 2) for k in METRICS} if ok else None,
            "old_judge": d.get("judge_model"),
            "old_delta": d.get("delta"),
        })
        if ok:
            avg = sum(res[-1]["delta"].values()) / len(METRICS)
            old = sum(d["delta"].values()) / len(METRICS) if d.get("delta") else float("nan")
            print(f"  delta avg {avg:+.2f} (old judge {old:+.2f})")
        else:
            print("  judge output not parseable, marked invalid")
        # save after every scenario so a crash or rate limit doesnt lose the work
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)

    ok = [r for r in res if r["valid"]]
    if not ok:
        print("no valid scenarios")
        return
    short = ["Coh.", "Cpl.", "Hum.", "Tec.", "Str.", "Ctx."]
    lines = [f"# Re-judge with `{config.JUDGE_MODEL}` (valid {len(ok)}/{len(res)})", "",
             "| Metric | Baseline | PsyPARSE | Δ | Δ old judge |", "| :--- | :---: | :---: | :---: | :---: |"]
    for k, s in zip(METRICS, short):
        bm = sum(r["baseline_scores"][k] for r in ok) / len(ok)
        pm = sum(r["psyparse_scores"][k] for r in ok) / len(ok)
        olds = [r["old_delta"][k] for r in ok if r["old_delta"]]
        om = f"{sum(olds) / len(olds):+.2f}" if olds else "-"
        lines.append(f"| {s} | {bm:.2f} | {pm:.2f} | {pm - bm:+.2f} | {om} |")
    wins = sum(1 for r in ok if sum(r["delta"].values()) > 0)
    lines += ["", f"PsyPARSE better on average in {wins}/{len(ok)} scenarios."]
    txt = "\n".join(lines)
    with open(config.RESULTS_DIR / f"rejudge_{name}.md", "w") as fh:
        fh.write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()

import re
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import config
from psyparse.retrieval.hybrid_search import HybridRetriever
from psyparse.pipeline.run_full import run_psyparse_pipeline
from psyparse.agents.base_agent import BaseAgent, clean_history, format_dialogue, has_cjk
from psyparse.agents.patient_agent import PatientAgent, get_opening, get_persona

logger = logging.getLogger(__name__)

METRICS = ["coherence", "completeness", "humaneness", "technique", "structure", "context_retention"]
JUDGE_PROMPT = (Path(__file__).resolve().parent / "judge_prompt.txt").read_text(encoding="utf-8")


def make_judge() -> BaseAgent:
    return BaseAgent(
        system_prompt="You are a strict, expert clinical psychology supervisor. You must respond ONLY with a valid JSON object, with no conversational text.",
        model=config.JUDGE_MODEL,
        temperature=config.TEMP_JUDGE,
        base_url=config.JUDGE_BASE_URL,
        api_key=config.JUDGE_API_KEY,
        reasoning_effort=config.JUDGE_REASONING_EFFORT,
    )


def judge_transcript(judge: BaseAgent, transcript_str: str, samples: int = config.JUDGE_SAMPLES) -> Optional[Dict[str, float]]:
    # average a few judge calls, returns None if the judge never gives parseable scores
    prompt = JUDGE_PROMPT.replace("{TRANSCRIPT}", transcript_str)
    # ~3 chars per token is on the safe side. ollama cuts the START of a too long prompt = the rubric, without any error
    if len(prompt) / 3 > config.CTX_LIMIT - 400:
        print(f"  [warn] judge prompt is ~{len(prompt) // 3} tokens, may not fit CTX_LIMIT={config.CTX_LIMIT}: raise the context length")
    runs = []
    tries = 0
    while len(runs) < samples and tries < samples + 3:
        tries += 1
        raw = judge.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=config.TEMP_JUDGE,
            response_format={"type": "json_object"},
        )
        try:
            txt = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE)
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            data = json.loads(m.group(0) if m else txt)
            runs.append({k: min(10.0, max(1.0, float(data[k]))) for k in METRICS})
        except Exception as e:
            logger.warning("judge output not parseable (%s): %s", e, raw[:200])
    if not runs:
        return None
    # report on a 10-100 scale like the paper
    return {k: round(sum(r[k] for r in runs) / len(runs) * 10.0, 2) for k in METRICS}


class EvaluatorHarness:
    def __init__(self, retriever: HybridRetriever, model: Optional[str] = None):
        self.retriever = retriever
        self.model = model or config.DEEPSEEK_MODEL
        self.judge_agent = make_judge()

    def run_baseline_dialogue(self, scenario: Dict[str, Any], max_turns: int = config.BASELINE_TURNS) -> List[Dict[str, str]]:
        # vanilla llm counselor: no topic, no background, no intake script (SOP: plain LLM baseline)
        therapist = BaseAgent(
            system_prompt="You are a professional psychological counselor. Have a supportive counseling conversation with the client.",
            model=self.model,
            temperature=0.7,
        )
        # same patient setup and same ending rule as the psyparse arm (intake + min therapy turns)
        patient = PatientAgent(scenario=scenario, model=self.model, temperature=0.7,
                               min_turns=config.INTAKE_TURNS + config.MIN_THERAPY_TURNS)
        opening = get_opening(scenario, model=self.model)
        patient.add_message("assistant", opening)

        print(f"  -> Simulating Baseline Dialogue (Vanilla LLM, {max_turns} turns)...")
        last = opening
        for turn in range(max_turns):
            therapist_msg = therapist.send(last)
            last = patient.respond(therapist_msg)
            if "[SESSION_END]" in last:
                break
            last = last.strip()
        print("  -> Baseline Dialogue Complete.")
        return patient.get_history()

    def evaluate_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        sid = scenario.get("dialog_id", "unknown")
        baseline = self.run_baseline_dialogue(scenario)
        psy = run_psyparse_pipeline(scenario, self.retriever, model=self.model)

        # judge each transcript on its own, speakers labeled Therapist/Patient, no arm-revealing tags
        print("  -> Judging transcripts...")
        b_scores = judge_transcript(self.judge_agent, format_dialogue(baseline))
        p_scores = judge_transcript(self.judge_agent, format_dialogue(psy["transcript"]))
        valid = b_scores is not None and p_scores is not None

        res = {
            "scenario_id": sid,
            "topic": scenario.get("topic", "General"),
            "gold_therapy": scenario.get("psychotherapy", ""),
            "best_therapy": psy["best_therapy"],
            "top_therapies": [t["therapy_type"] for t in psy["top_therapies"]],
            "agent_model": self.model,
            "judge_model": config.JUDGE_MODEL,
            "quick_test": config.QUICK_TEST,
            "baseline_turns": len([m for m in baseline if m["role"] == "user"]),
            "psyparse_turns": psy["turns"],
            "valid": valid,
            # should stay empty now that the patient retries in english, kept to spot it if it still happens
            "non_english": [n for n, h in [("baseline", baseline), ("psyparse", psy["transcript"])]
                            if any(has_cjk(m["content"]) for m in h)],
            "baseline_scores": b_scores,
            "psyparse_scores": p_scores,
            "delta": {k: round(p_scores[k] - b_scores[k], 2) for k in METRICS} if valid else None,
        }

        # full record for manual review (SOP E2)
        tdir = config.RESULTS_DIR / "transcripts"
        tdir.mkdir(parents=True, exist_ok=True)
        detail = dict(res)
        detail.update({
            "patient_persona": get_persona(scenario, self.model),
            "patient_profile": psy["patient_profile"],
            "keywords": psy["keywords"],
            "top_therapies_detail": psy["top_therapies"],
            "guidance": psy["guidance"],
            "rollout_logs": psy["rollout_logs"],
            "pruning_logs": psy["pruning_logs"],
            "baseline_transcript": clean_history(baseline),
            "psyparse_transcript": clean_history(psy["transcript"]),
        })
        with open(tdir / f"{sid}.json", "w", encoding="utf-8") as f:
            json.dump(detail, f, indent=2)
        return res


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    resume = "--resume" in sys.argv
    n_scenarios = int(args[0]) if args else 3

    try:
        with open(config.DATA_DIR / "eval_scenarios.json", "r", encoding="utf-8") as f:
            scenarios = json.load(f)
    except FileNotFoundError:
        print("Error: data/eval_scenarios.json not found.")
        return
    scenarios = scenarios[:n_scenarios]

    config.RESULTS_DIR.mkdir(exist_ok=True)
    out_path = config.RESULTS_DIR / "eval_results.json"
    results = []
    if resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            results = [r for r in json.load(f) if "valid" in r]
        print(f"Resuming, {len(results)} scenarios already done")
    done = {r["scenario_id"] for r in results}

    if config.QUICK_TEST:
        print("*** QUICK_TEST mode: tiny settings to check the pipeline works, scores are NOT real results ***")
    print(f"agent model = {config.DEEPSEEK_MODEL} | judge model = {config.JUDGE_MODEL} | "
          f"turns: intake {config.INTAKE_TURNS} + therapy {config.THERAPY_TURNS}, baseline {config.BASELINE_TURNS} | "
          f"rounds {config.ROLLOUT_ROUNDS}, branches {config.PRUNING_BRANCHES}, k2 {config.RAG_TOP_K2}, "
          f"parallel {config.PARALLEL_CALLS}, results -> {config.RESULTS_DIR.name}/")
    print("Initializing HybridRetriever...")
    retriever = HybridRetriever()
    harness = EvaluatorHarness(retriever=retriever)

    for i, scenario in enumerate(scenarios):
        if scenario.get("dialog_id") in done:
            continue
        print(f"\nEvaluating scenario {i+1}/{len(scenarios)} ({scenario.get('dialog_id')}, {scenario.get('topic')})...")
        res = harness.evaluate_scenario(scenario)
        print(f"  delta: {res['delta']}")
        results.append(res)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"\nEvaluation complete! Results saved to {out_path}")
    from evaluation.aggregate import aggregate_eval_results
    print(aggregate_eval_results(str(out_path)))


if __name__ == "__main__":
    main()

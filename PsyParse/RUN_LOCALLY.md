# Running PsyParse Locally (Ollama)

Everything runs on your own machine with [Ollama](https://ollama.com/). There are no API limits, but speed depends on your GPU.

## 1. Install
All commands run from the **inner `PsyParse` folder** (the one that contains `config.py`).
- Fresh clone: `cd PsyParse/PsyParse`.
- Existing copy (e.g. `~/aa_ML/Project`): `cd PsyParse`.

```bash
git clone https://github.com/manishsn7340/PsyParse.git
cd PsyParse/PsyParse              # existing copy: cd PsyParse

python3 -m venv venv              # skip if this folder already has a venv/
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Start Ollama (context size and parallel calls matter)
```bash
OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```
- **`OLLAMA_CONTEXT_LENGTH=16384`:** Ollama's default context is only 2k–4k tokens, and it **silently cuts off the start** of longer prompts. A 15-turn transcript plus the judge rubric can reach ~10.5k tokens, so 16384 leaves room (the longest baseline transcript reached ~10.5k).
- **`OLLAMA_NUM_PARALLEL=4`:** lets the 4 candidate replies, scores and rollouts run at the same time. More parallel slots need more GPU memory: use `2` with a 32B model, or `1` if a model doesn't fit.

## 3. Pull models
All Ollama models are already **4-bit quantized**, so there is nothing more to quantize. Bigger models are smarter but slower.

| Your GPU memory | Agents (`DEEPSEEK_MODEL`) | Judge (`JUDGE_MODEL`) |
|---|---|---|
| 8–12 GB | `qwen2.5:7b` | `qwen2.5:14b`, or a Groq judge (see step 4) |
| 16 GB | `qwen2.5:7b` or `qwen2.5:14b` | `qwen2.5:14b`, or a Groq judge |
| 24 GB+ | `qwen2.5:14b` | `qwen2.5:32b` |

```bash
ollama pull qwen2.5:14b
ollama pull qwen2.5:32b
```
The judge should be **bigger than the agents**. If the judge is the same small model, it gives 90+ to everything and the deltas mean nothing.

## 4. Create `.env` (in the inner `PsyParse` folder, next to `config.py`)
```env
DEEPSEEK_API_KEY=ollama
DEEPSEEK_BASE_URL=http://localhost:11434/v1
DEEPSEEK_MODEL=qwen2.5:14b
JUDGE_MODEL=qwen2.5:32b

# match the ollama server settings from step 2
PARALLEL_CALLS=4
CTX_LIMIT=16384

# optional
JUDGE_SAMPLES=3      # judge runs averaged per transcript
THERAPY_TURNS=5      # 5 = quicker runs (baseline 10 turns); final run uses 10 (baseline 15, as in the SOP)
# MAX_TOKENS=1024    # output cap per call
```

**Option: big judge on Groq, agents on Ollama.** Use this if your GPU can't fit a big judge. With `JUDGE_SAMPLES=1` the judge makes only 2 calls per scenario, so the Groq free tier is enough for it:
```env
JUDGE_MODEL=openai/gpt-oss-120b
JUDGE_BASE_URL=https://api.groq.com/openai/v1
JUDGE_API_KEY=your_groq_key
JUDGE_SAMPLES=1
JUDGE_REASONING_EFFORT=low   # gpt-oss is a thinking model, keep its thinking short
```

## 5. Run
Always run from the inner `PsyParse` folder (next to `config.py`) with `PYTHONPATH=.`. Shell variables (like `QUICK_TEST=1` below) override `.env`.

**Step 1: quick test (minutes).** Checks that everything works end to end with tiny settings: 3 intake + 2 therapy turns, 1 rollout round, 2 branches, 1 judge run (~40 LLM calls per scenario instead of 150+). It writes to `results_quick/`. **These numbers are not results.**
```bash
ollama pull qwen2.5:7b     # small agent model, only needed for quick tests
QUICK_TEST=1 DEEPSEEK_MODEL=qwen2.5:7b JUDGE_MODEL=qwen2.5:14b PYTHONPATH=. python evaluation/run_eval.py 2
```
Then open `results_quick/transcripts/<id>.json` and check:
- speakers are labeled correctly (Therapist / Patient)
- the therapist uses details from the intake
- the therapy names match the retrieved ones

**Step 2: judge check.** The strong transcript must score clearly higher than the weak one (gap > 15). If not, use a bigger judge.
```bash
PYTHONPATH=. python test_judge.py
```

**Step 3: grid search for alpha / w1 / w2** (dev set; only the 4 intake interviews use the LLM, cached in `results/dev_profiles.json`).
```bash
PYTHONPATH=. python evaluation/grid_search.py
```
It prints `best: {'alpha': ..., 'w1': ..., 'w2': ...}`. Pass these values on the command line (or put them in `config.py`).

**Step 4: final run** (SOP: N ≥ 10 scenarios, 10 therapy turns).
```bash
THERAPY_TURNS=10 ALPHA=0.5 W1=0.5 W2=0.5 PYTHONPATH=. python evaluation/run_eval.py 10 --resume
```
`--resume` skips scenarios already saved in `results/eval_results.json`, so you can stop and continue later. Results are saved after every scenario.

**Other commands**
```bash
PYTHONPATH=. python evaluation/aggregate.py           # rebuild results/summary_table.md
PYTHONPATH=. python -m psyparse.pipeline.run_full     # demo: one dev scenario, printed + saved to test_outputs/
```

## Outputs
- `results/eval_results.json`: scores and deltas per scenario
- `results/summary_table.md`: means, Δ and SOP targets
- `results/transcripts/<id>.json`: both transcripts, profile, retrieved therapies, chosen therapy, rollout and pruning scores (for the manual review)
- `results_quick/`: the same files for quick-test runs (not real results)

## Troubleshooting
- **`[warn] a call used ... tokens, close to CTX_LIMIT`:** restart Ollama with `OLLAMA_CONTEXT_LENGTH=24576` and set `CTX_LIMIT=24576`. Lower `OLLAMA_NUM_PARALLEL` if memory gets tight.
- **`[warn] a reply hit max_tokens`:** set `MAX_TOKENS=1536` in `.env`.
- **Very slow, or the GPU isn't used:** the model doesn't fit and Ollama moved part of it to the CPU (check `ollama ps`). Use smaller models or lower `OLLAMA_NUM_PARALLEL`.
- **`[warn] profile extraction failed`:** the agent model returned bad JSON. A bigger agent model usually fixes it.

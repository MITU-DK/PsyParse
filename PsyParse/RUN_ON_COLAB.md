# Running PsyParse on Google Colab / Kaggle (Ollama)

The notebook clones the GitHub repo, so **push your latest code first**. Colab/Kaggle only see what is on GitHub.

**GPU settings**

- **Kaggle:** right sidebar → **Accelerator: GPU T4 x2** (2 × 15 GB) and **Internet: On**.
- **Colab:** `Runtime → Change runtime type → T4 GPU` (1 × 15 GB). Colab Pro gives L4 (24 GB) or A100 (40 GB).

The setup cell detects the platform and GPU memory and picks models that fit. All Ollama models are already 4-bit, so there is nothing more to quantize. Speed comes from smaller models, fewer calls (quick mode) and parallel calls.

| MODE      | Agents      | Judge                                                                         | What it runs                                                                                                  | Use it for                                                                                                 |
| --------- | ----------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `"quick"` | qwen2.5:7b  | qwen2.5:14b                                                                   | `QUICK_TEST=1`: 3 intake + 2 therapy turns, 1 rollout round, 2 branches, 1 judge run (~40 LLM calls/scenario) | finding errors and reading a transcript in minutes. **Numbers are not results**; saved to `results_quick/` |
| `"full"`  | qwen2.5:14b | qwen2.5:32b if ≥ 28 GB GPU memory (Kaggle T4 x2, A100), otherwise qwen2.5:14b | paper/SOP settings (~150+ LLM calls/scenario)                                                                 | real results in `results/`                                                                                 |

On a single T4 (free Colab) the full-mode judge is only as big as the agents, which makes it weak. Better: set `GROQ_JUDGE_KEY` in cell 1. The judge then runs on Groq's `openai/gpt-oss-120b` (1 call per transcript, so the free tier is enough for ~10 scenarios a day) while the agents stay on the GPU.

---

## Cell 1: setup (run once per session; to change MODE, restart the session)

```python
import os, subprocess, time, urllib.request

MODE = "full"           # "quick" = smoke test only, "full" = real results (see the table below)
GROQ_JUDGE_KEY = ""     # optional: Groq API key -> big judge on Groq, agents stay on the GPU

WORKDIR = "/kaggle/working" if os.path.exists("/kaggle") else "/content"
os.chdir(WORKDIR)

# total gpu memory in GB (2x T4 on kaggle = ~30)
out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
GPU_GB = sum(int(x) for x in out.split()) / 1024
print(f"platform: {WORKDIR} | GPU memory: {GPU_GB:.0f} GB")

if MODE == "quick":
    AGENT_MODEL, JUDGE_MODEL = "qwen2.5:7b", "qwen2.5:14b"
else:
    AGENT_MODEL = "qwen2.5:14b"
    JUDGE_MODEL = "qwen2.5:32b" if GPU_GB >= 28 else "qwen2.5:14b"
if GROQ_JUDGE_KEY:
    JUDGE_MODEL = "openai/gpt-oss-120b"
# more parallel slots = faster but more gpu memory; 32b or a single 15GB gpu only fits 2
N_PARALLEL = "4" if GPU_GB >= 28 and JUDGE_MODEL != "qwen2.5:32b" else "2"
CTX = "16384"           # long 15-turn baseline transcripts + rubric reach ~10.5k tokens; ollama silently cuts anything longer

!apt-get update -qq && apt-get install -y -qq zstd
!curl -fsSL https://ollama.com/install.sh | sh

# flash attention + 8-bit kv cache halve the context memory, so the 32b judge with a 12k context stays fully on the gpus
env = dict(os.environ, OLLAMA_CONTEXT_LENGTH=CTX, OLLAMA_NUM_PARALLEL=N_PARALLEL, OLLAMA_KEEP_ALIVE="30m",
           OLLAMA_FLASH_ATTENTION="1", OLLAMA_KV_CACHE_TYPE="q8_0")
# a restarted notebook keeps the old server running, and then the new settings above would be silently ignored
!pkill -f "ollama serve"; sleep 3
subprocess.Popen(["ollama", "serve"], env=env, stdout=open(f"{WORKDIR}/ollama.log", "w"), stderr=subprocess.STDOUT)
for _ in range(30):
    try:
        urllib.request.urlopen("http://localhost:11434"); break
    except Exception:
        time.sleep(2)

!ollama pull {AGENT_MODEL}
if not GROQ_JUDGE_KEY and JUDGE_MODEL != AGENT_MODEL:
    !ollama pull {JUDGE_MODEL}

# keep results from an earlier (crashed) run before the fresh clone, see "Time limits" below
!mkdir -p results_backup && cp -r PsyParse/PsyParse/results "results_backup/results_$(date +%H%M%S)" 2>/dev/null; true
!rm -rf PsyParse
!git clone https://github.com/manishsn7340/PsyParse.git
!pip install -q -r PsyParse/PsyParse/requirements.txt

with open("PsyParse/PsyParse/.env", "w") as f:
    f.write("DEEPSEEK_API_KEY=ollama\n")
    f.write("DEEPSEEK_BASE_URL=http://localhost:11434/v1\n")
    f.write(f"DEEPSEEK_MODEL={AGENT_MODEL}\n")
    f.write(f"JUDGE_MODEL={JUDGE_MODEL}\n")
    if GROQ_JUDGE_KEY:
        f.write("JUDGE_BASE_URL=https://api.groq.com/openai/v1\n")
        f.write(f"JUDGE_API_KEY={GROQ_JUDGE_KEY}\n")
        f.write("JUDGE_SAMPLES=1\n")   # keeps the judge inside groq's free daily token limit
        f.write("JUDGE_REASONING_EFFORT=low\n")   # gpt-oss is a thinking model, keep its thinking short
    f.write(f"PARALLEL_CALLS={N_PARALLEL}\n")
    f.write(f"CTX_LIMIT={CTX}\n")
    f.write("QUICK_TEST=1\n" if MODE == "quick" else "QUICK_TEST=0\n")

%cd {WORKDIR}/PsyParse/PsyParse
print(f"Setup complete | mode={MODE} agents={AGENT_MODEL} judge={JUDGE_MODEL} parallel={N_PARALLEL}")
```

## How to use the cells

Two separate sessions. MODE is fixed by Cell 1 and written to `.env`, so **every later cell uses the MODE from Cell 1**. Changing a command later does not switch modes.

| Session                           | Cell 1 with      | Then run                                                        | Output                         |
| --------------------------------- | ---------------- | --------------------------------------------------------------- | ------------------------------ |
| A: smoke test (optional, ~20 min) | `MODE = "quick"` | Cell 2, then `!PYTHONPATH=. python evaluation/run_eval.py 1`, then **stop** | `results_quick/` (not results) |
| B: real results (hours)           | `MODE = "full"`  | Cell 2, Cell 3, Cell 4 (1 scenario, send it for a check), then Cell 5, Cell 6 | `results/`                     |

To go from session A to B: **Run → Restart & clear cell outputs** (Kaggle) / **Runtime → Restart session** (Colab), set `MODE = "full"` in Cell 1, and run Cell 1 again. That pulls the bigger models and writes `QUICK_TEST=0`. Cells 3, 4 and 5 refuse to run if Cell 1 was not in full mode.

Check the first lines that `run_eval.py` prints:

- quick: `*** QUICK_TEST mode ***` ... `intake 3 + therapy 2, baseline 5 | rounds 1, branches 2, k2 2` ... `results -> results_quick/`
- full (Cells 4 and 5): `intake 5 + therapy 10, baseline 15 | rounds 3, branches 4, k2 3` ... `results -> results/`

## Cell 2: check the judge (a few minutes, both sessions)

The strong transcript must score clearly higher than the weak one (gap > 15). If not, the judge is too weak and the final deltas won't mean much; use the Groq judge option.

```python
!PYTHONPATH=. python test_judge.py
```

## Cell 3: grid search for alpha / w1 / w2 (session B only, dev set, 4 interviews)

```python
assert MODE == "full", "restart the session and run Cell 1 with MODE = 'full' first"
!QUICK_TEST=0 PYTHONPATH=. python evaluation/grid_search.py
```

It prints `best: {'alpha': ..., 'w1': ..., 'w2': ...}`. Write those three numbers into Cells 4 and 5 (no need to edit `config.py`).

## Cell 4: check run, 1 scenario first (session B only, ~30-40 min)

Runs only the first scenario with the real models, so a bad setup shows up before the long run. Send `results/transcripts/000432.json` (or zip `results/`) for a check: are the PsyParse intake and therapy replies about as long and as concrete as the baseline's?

```python
assert MODE == "full", "restart the session and run Cell 1 with MODE = 'full' first"
ALPHA, W1, W2 = 0.5, 0.5, 0.5     # <- the best values from Cell 3
!QUICK_TEST=0 THERAPY_TURNS=10 ALPHA={ALPHA} W1={W1} W2={W2} PYTHONPATH=. python evaluation/run_eval.py 1
```

## Cell 5: final run (session B only, SOP: N ≥ 10, 10 therapy turns)

Same command with 10 scenarios. `--resume` keeps the scenario from Cell 4 and continues with the other 9.

```python
assert MODE == "full", "restart the session and run Cell 1 with MODE = 'full' first"
ALPHA, W1, W2 = 0.5, 0.5, 0.5     # <- the best values from Cell 3
!QUICK_TEST=0 THERAPY_TURNS=10 ALPHA={ALPHA} W1={W1} W2={W2} PYTHONPATH=. python evaluation/run_eval.py 10 --resume
```

`--resume` skips scenarios already saved in `results/eval_results.json`. Results are saved after every scenario, so if the session stops you lose at most one scenario.

## Cell 6: download results

```python
!zip -qr results.zip results
!cat results/summary_table.md
```

- **Kaggle:** download `results.zip` from the Output panel (`/kaggle/working`). Kaggle may need a page refresh before the file shows up.
- **Colab:** `from google.colab import files; files.download("results.zip")`.

---

### Notes

- **Run stopped but the session is still alive** (kernel crash, you pressed stop): run Cell 1 again (it copies the old results to `/kaggle/working/results_backup/` before re-cloning), then restore them and run Cell 5 again (it continues with `--resume`):
  ```python
  !cp -rT "$(ls -d /kaggle/working/results_backup/results_* | tail -1)" results
  ```
- Keep the browser tab open during the long run; an idle/closed interactive session can be stopped by Kaggle.
- **Time limits:** Kaggle sessions stop after 12 hours (~30 GPU hours per week). Free Colab disconnects after a few hours or when idle. If a full run won't fit, run `run_eval.py 5`, download, then continue in a new session: run Cell 1 with `MODE = "full"`, upload `eval_results.json` into `results/` (create the folder first: `!mkdir -p results`), and run Cell 5 again with the same ALPHA/W1/W2 and `--resume`.
- If you see `[warn] a call used ... tokens, close to CTX_LIMIT`: set `CTX = "24576"` in Cell 1 and `N_PARALLEL = "1"` (more context per slot uses more GPU memory).
- If you see `[warn] a reply hit max_tokens`: add `f.write("MAX_TOKENS=1536\n")` to the `.env` lines in Cell 1.
- **Very slow:** the model may not fit in GPU memory and part of it moved to the CPU. Run `!ollama ps` while the judge is working: the PROCESSOR column must say `100% GPU`. If it shows a CPU share, set `N_PARALLEL = "1"` in Cell 1 and restart.
- **Groq judge errors (429):** the free daily token limit was reached. Wait, or lower `JUDGE_SAMPLES` (add `f.write("JUDGE_SAMPLES=1\n")`).

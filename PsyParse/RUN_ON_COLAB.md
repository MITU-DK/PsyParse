# Running PsyParse on Google Colab / Kaggle (Ollama)

The notebook clones the GitHub repo, so **push your latest code first**. Colab/Kaggle only see what is on GitHub.

**GPU settings**
- **Kaggle:** right sidebar → **Accelerator: GPU T4 x2** (2 × 15 GB) and **Internet: On**.
- **Colab:** `Runtime → Change runtime type → T4 GPU` (1 × 15 GB). Colab Pro gives L4 (24 GB) or A100 (40 GB).

The setup cell detects the platform and GPU memory and picks models that fit. All Ollama models are already 4-bit, so there is nothing more to quantize. Speed comes from smaller models, fewer calls (quick mode) and parallel calls.

| MODE | Agents | Judge | What it runs | Use it for |
|---|---|---|---|---|
| `"quick"` | qwen2.5:7b | qwen2.5:14b | `QUICK_TEST=1`: 3 intake + 2 therapy turns, 1 rollout round, 2 branches, 1 judge run (~40 LLM calls/scenario) | finding errors and reading a transcript in minutes. **Numbers are not results**; saved to `results_quick/` |
| `"full"` | qwen2.5:14b | qwen2.5:32b if ≥ 28 GB GPU memory (Kaggle T4 x2, A100), otherwise qwen2.5:14b | paper/SOP settings (~150+ LLM calls/scenario) | real results in `results/` |

On a single T4 (free Colab) the full-mode judge is only as big as the agents, which makes it weak. Better: set `GROQ_JUDGE_KEY` in cell 1. The judge then runs on Groq's `openai/gpt-oss-120b` (1 call per transcript, so the free tier is enough for ~10 scenarios a day) while the agents stay on the GPU.

---

## Cell 1: setup (run once per session; to change MODE, restart the session)

```python
import os, subprocess, time, urllib.request

MODE = "quick"          # "quick" or "full"
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
CTX = "8192"            # enough for 15-turn transcripts; ollama silently cuts anything longer

!apt-get update -qq && apt-get install -y -qq zstd
!curl -fsSL https://ollama.com/install.sh | sh

env = dict(os.environ, OLLAMA_CONTEXT_LENGTH=CTX, OLLAMA_NUM_PARALLEL=N_PARALLEL, OLLAMA_KEEP_ALIVE="30m")
subprocess.Popen(["ollama", "serve"], env=env, stdout=open(f"{WORKDIR}/ollama.log", "w"), stderr=subprocess.STDOUT)
for _ in range(30):
    try:
        urllib.request.urlopen("http://localhost:11434"); break
    except Exception:
        time.sleep(2)

!ollama pull {AGENT_MODEL}
if not GROQ_JUDGE_KEY and JUDGE_MODEL != AGENT_MODEL:
    !ollama pull {JUDGE_MODEL}

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

## Cell 2: check the judge (a few minutes)
The strong transcript must score clearly higher than the weak one (gap > 15). If not, the judge is too weak and the final deltas won't mean much; use the Groq judge option.
```python
!PYTHONPATH=. python test_judge.py
```

## Cell 3: quick end-to-end test (MODE = "quick")
```python
!PYTHONPATH=. python evaluation/run_eval.py 2
!cat results_quick/summary_table.md
```
Open `results_quick/transcripts/<id>.json` (file browser on the left) and check:
- speakers are labeled correctly (Therapist / Patient)
- the therapist uses details from the intake
- the therapy names match
- nothing looks broken

The scores here mean nothing; this only checks that everything works.

## Cell 4: grid search for alpha / w1 / w2 (MODE = "full", dev set, 4 interviews)
```python
!PYTHONPATH=. python evaluation/grid_search.py
```
It prints `best: {'alpha': ..., 'w1': ..., 'w2': ...}`. Put those numbers into the next cell (no need to edit `config.py`).

## Cell 5: final run (MODE = "full", SOP: N ≥ 10, 10 therapy turns)
```python
!THERAPY_TURNS=10 ALPHA=0.5 W1=0.5 W2=0.5 PYTHONPATH=. python evaluation/run_eval.py 10 --resume
```
`--resume` skips scenarios already saved in `results/eval_results.json`. Results are saved after every scenario.

## Cell 6: download results
```python
!zip -qr results.zip results results_quick
!cat results/summary_table.md
```
- **Kaggle:** download `results.zip` from the Output panel (`/kaggle/working`).
- **Colab:** `from google.colab import files; files.download("results.zip")`.

---

### Notes
- **Time limits:** Kaggle sessions stop after 12 hours (~30 GPU hours per week). Free Colab disconnects after a few hours or when idle. If a full run won't fit, run `run_eval.py 5`, download, then continue in a new session: run Cell 1, upload `eval_results.json` into `results/`, and run Cell 5 again with `--resume`.
- If you see `[warn] a call used ... tokens, close to CTX_LIMIT`: set `CTX = "16384"` in Cell 1 and `N_PARALLEL = "1"` (more context per slot uses more GPU memory).
- If you see `[warn] a reply hit max_tokens`: add `f.write("MAX_TOKENS=800\n")` to the `.env` lines in Cell 1.
- **Very slow:** the model may not fit in GPU memory and part of it moved to the CPU. Check `!ollama ps` and `ollama.log`, then use smaller models or a lower `N_PARALLEL`.
- **Groq judge errors (429):** the free daily token limit was reached. Wait, or lower `JUDGE_SAMPLES` (add `f.write("JUDGE_SAMPLES=1\n")`).

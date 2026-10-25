# Running PsyParse with Groq Cloud (API)

Groq runs the models in the cloud, so **no GPU is needed** (works locally, or on Kaggle/Colab with CPU only), and it's much faster than a T4. The catch is the **free-tier limits**: tokens per minute and tokens per day per model. Check your current limits at [console.groq.com/settings/limits](https://console.groq.com/settings/limits).

What fits on the free tier:
- **Quick tests (`QUICK_TEST=1`, ~40 calls/scenario): yes.** Best use of Groq.
- **Judge only, with agents on Ollama: yes.** See `RUN_LOCALLY.md` / `RUN_ON_COLAB.md`, the "Groq judge" option. The judge makes only a few calls per scenario.
- **The full run (~150+ calls/scenario × 10 scenarios): probably not in one day.** The daily token limit of the small model runs out after a few full scenarios. Use `--resume` and spread it over several days, or run the agents on Ollama.

**Models** (Groq has retired the Llama chat models; check the current list with the cell below)
- **Agents** (therapist / patient / evaluator): `openai/gpt-oss-20b`
- **Judge:** `openai/gpt-oss-120b`. It has to be bigger than the agents, otherwise it gives 90+ to everything.
- Both are "thinking" models: their hidden reasoning counts against the output limit. Always set **`REASONING_EFFORT=low`** (short thinking, faster) and **`MAX_TOKENS=2048`** (room for thinking + reply). If you see `[warn] a reply hit max_tokens`, raise it to 3072.
- Groq models have a large context (131k), so set `CTX_LIMIT=131072`. The 16384 default is for Ollama.

Always set **`PARALLEL_CALLS=1`** with Groq. Parallel calls just hit the per-minute limit (429 errors). The code retries 429s automatically with a growing wait, so a run slows down instead of crashing.

---

## Option A: Kaggle / Colab notebook (no GPU needed)

**Push your latest code to GitHub first**, because the notebook clones it.

### Cell 1: setup
```python
import os, json, urllib.request

API_KEY = "YOUR_GROQ_API_KEY_HERE"
AGENT_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "openai/gpt-oss-120b"
QUICK = True            # True = quick test settings (recommended on the free tier)

WORKDIR = "/kaggle/working" if os.path.exists("/kaggle") else "/content"
os.chdir(WORKDIR)

# check the models still exist (groq renames/retires models sometimes)
req = urllib.request.Request("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {API_KEY}"})
available = [m["id"] for m in json.loads(urllib.request.urlopen(req).read().decode())["data"]]
for m in (AGENT_MODEL, JUDGE_MODEL):
    print(("OK      " if m in available else "MISSING ") + m)
print("available models:", sorted(available))

!rm -rf PsyParse
!git clone https://github.com/manishsn7340/PsyParse.git
!pip install -q -r PsyParse/PsyParse/requirements.txt

with open("PsyParse/PsyParse/.env", "w") as f:
    f.write(f"DEEPSEEK_API_KEY={API_KEY}\n")
    f.write("DEEPSEEK_BASE_URL=https://api.groq.com/openai/v1\n")
    f.write(f"DEEPSEEK_MODEL={AGENT_MODEL}\n")
    f.write(f"JUDGE_MODEL={JUDGE_MODEL}\n")
    f.write("PARALLEL_CALLS=1\n")
    f.write("REASONING_EFFORT=low\n")
    f.write("MAX_TOKENS=2048\n")
    f.write("CTX_LIMIT=131072\n")
    f.write("JUDGE_SAMPLES=1\n" if QUICK else "JUDGE_SAMPLES=2\n")
    f.write("QUICK_TEST=1\n" if QUICK else "QUICK_TEST=0\n")

%cd {WORKDIR}/PsyParse/PsyParse
print("Groq setup complete")
```
If a model shows `MISSING`, pick a similar one from the printed list (a smaller model for the agents, a bigger one for the judge).

### Cell 2: check the judge
The strong transcript must score clearly higher than the weak one (gap > 15).
```python
!PYTHONPATH=. python test_judge.py
```

### Cell 3: quick end-to-end test (`QUICK = True`)
```python
!PYTHONPATH=. python evaluation/run_eval.py 2
!cat results_quick/summary_table.md
```
Open `results_quick/transcripts/<id>.json` and check:
- the speakers are labeled correctly
- the therapist uses intake details
- the therapy names match

These numbers are **not** results.

### Cell 4: grid search (`QUICK = False`, cheap: only 4 intake interviews)
```python
!PYTHONPATH=. python evaluation/grid_search.py
```

### Cell 5: full run (`QUICK = False`, use the best alpha/w1/w2 from Cell 4)
```python
!THERAPY_TURNS=10 ALPHA=0.5 W1=0.5 W2=0.5 PYTHONPATH=. python evaluation/run_eval.py 10 --resume
```
If the daily limit runs out, the run keeps waiting and retrying. Stop it, download `results/`, and continue another day: run Cell 1, upload `eval_results.json` into `results/`, and run Cell 5 again with `--resume`.

### Cell 6: download
```python
!zip -qr results.zip results results_quick
```
- **Kaggle:** download `results.zip` from the Output panel.
- **Colab:** `from google.colab import files; files.download("results.zip")`.

---

## Option B: on your own machine

Run from the inner `PsyParse` folder (the one with `config.py`). After a fresh clone that's `PsyParse/PsyParse`; in an existing copy like `~/aa_ML/Project` it's just `PsyParse`.
```bash
cd PsyParse/PsyParse              # existing copy: cd PsyParse
python3 -m venv venv && source venv/bin/activate   # skip venv creation if venv/ already exists
pip install -r requirements.txt
```

`.env` (in the inner `PsyParse` folder, next to `config.py`; it is gitignored, so your key stays private):
```env
DEEPSEEK_API_KEY=your_groq_key
DEEPSEEK_BASE_URL=https://api.groq.com/openai/v1
DEEPSEEK_MODEL=openai/gpt-oss-20b
JUDGE_MODEL=openai/gpt-oss-120b
PARALLEL_CALLS=1
JUDGE_SAMPLES=1
REASONING_EFFORT=low
MAX_TOKENS=2048
CTX_LIMIT=131072
```

Then the same commands (from the inner `PsyParse` folder):
```bash
PYTHONPATH=. python test_judge.py
QUICK_TEST=1 PYTHONPATH=. python evaluation/run_eval.py 2
PYTHONPATH=. python evaluation/grid_search.py
THERAPY_TURNS=10 PYTHONPATH=. python evaluation/run_eval.py 10 --resume
```

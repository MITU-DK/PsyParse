# Running PsyParse on Google Colab / Kaggle

Google Colab and Kaggle are great environments for running PsyParse. Because these environments reset every time you close them, you'll need to run a setup block first to get everything installed.

## ⚠️ GPU Constraints to know
- **Free Colab / Kaggle T4x2**: Gives you T4 GPU(s) (~15GB VRAM). This is perfect for running `llama3.1:8b`. It does *not* have enough memory for `qwen2.5:32b`.
- **Colab Pro**: Gives you A100 or L4 GPUs, which have plenty of VRAM (24GB-40GB) to easily run massive models.

---

## 1. The "Nuke from Orbit" Setup Cell
Open a new notebook, make sure you are using a GPU (`Runtime` -> `Change runtime type` -> `T4 GPU`), and paste this entire block into the very first cell. 

This will automatically install everything, start the server, clone your repository, pull the LLaMA 3.1 model, and set up the `.env` file in one shot:

```python
import subprocess, time, os

!apt-get update && apt-get install -y zstd
!curl -fsSL https://ollama.com/install.sh | sh

subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3)

# Pull the model (Switching to LLaMA 3.1 8B as discussed for better clinical reasoning and fast grid search)
!ollama pull llama3.1:8b

!rm -rf PsyParse
!git clone https://github.com/manishsn7340/PsyParse.git

# Install dependencies (ignoring faiss/torch errors if any)
!pip install -r PsyParse/PsyParse/requirements.txt faiss-cpu sentence-transformers rank_bm25

# Create .env in the nested folder
with open('PsyParse/PsyParse/.env', 'w') as f:
    f.write('DEEPSEEK_API_KEY=ollama\n')
    f.write('DEEPSEEK_BASE_URL=http://localhost:11434/v1\n')
    f.write('DEEPSEEK_MODEL=llama3.1:8b\n')

print("Setup Complete! Ready for evaluation.")
```

---

## 2. Train the Model (Run Grid Search)
Before evaluating, you should run a grid search to optimize the retrieval hyperparameter (`alpha`). LLaMA 3.1 8B is fast enough to do this quickly. Create a new cell and run:

```python
# For Colab use /content, for Kaggle use /kaggle/working
%cd /kaggle/working/PsyParse/PsyParse 
!PYTHONPATH=. python evaluation/grid_search.py
```
*(Note: If you are on Colab, change the path to `%cd /content/PsyParse/PsyParse`)*

---

## 3. Run the Evaluation
Once the optimal weights are found (or if you just want to test the pipeline), run the full evaluation to generate the final scores:

```python
!PYTHONPATH=. python evaluation/run_eval.py 1
```

---

**Downloading results**: When it finishes, click the little **Folder icon** on the far left sidebar of Kaggle/Colab. Navigate to `results`, hover over `eval_results.json` or `summary_table.md`, click the three dots, and select **Download**.

# Running PsyParse on Google Colab

Google Colab is a great alternative to Kaggle for running PsyParse. The setup is very similar, but because Colab environments reset every time you close them, you'll need to run a setup block first to get everything installed.

## ⚠️ GPU Constraints to know
- **Free Colab**: Gives you 1x T4 GPU (15GB VRAM). This is perfect for running `qwen2.5:14b` or `llama3.1:8b`. It does *not* have enough memory for `qwen2.5:32b`.
- **Colab Pro**: Gives you A100 or L4 GPUs, which have plenty of VRAM (24GB-40GB) to easily run massive models like `qwen2.5:32b`.

---

## 1. The "Nuke from Orbit" Setup Cell
Open a new Google Colab notebook, make sure you are using a GPU (`Runtime` -> `Change runtime type` -> `T4 GPU`), and paste this entire block into the very first cell. 

This will automatically install everything, start the server, clone your repository, pull the model, and set up the `.env` file in one shot:

```python
import subprocess, time, os

!apt-get update && apt-get install -y zstd
!curl -fsSL https://ollama.com/install.sh | sh

subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3)

# Pull the model (change to qwen2.5:32b if on Colab Pro)
!ollama pull qwen2.5:14b

!rm -rf PsyParse
!git clone https://github.com/manishsn7340/PsyParse.git

# THE FIX: The true path is the double folder!
!pip install -r PsyParse/PsyParse/requirements.txt

# Create .env in the nested folder
with open('PsyParse/PsyParse/.env', 'w') as f:
    f.write('DEEPSEEK_API_KEY=ollama\n')
    f.write('DEEPSEEK_BASE_URL=http://localhost:11434/v1\n')
    f.write('DEEPSEEK_MODEL=qwen2.5:14b\n')

print("Setup Complete! Ready for evaluation.")
```

---

## 2. Run the Evaluation
Once that first cell finishes and prints "Setup Complete!", create a second code cell below it and run your evaluation:

```python
%cd /kaggle/working/PsyParse/PsyParse
!PYTHONPATH=. python evaluation/run_eval.py 1
```

---


**Downloading results**: When it finishes, click the little **Folder icon** on the far left sidebar of Kaggle/Colab. Navigate to `results`, hover over `eval_results.json` or `eval_summary.md`, click the three dots, and select **Download**.

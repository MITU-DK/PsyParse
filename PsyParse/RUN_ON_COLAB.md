# Running PsyParse on Google Colab

Google Colab is a great alternative to Kaggle for running PsyParse. The setup is very similar, but because Colab environments reset every time you close them, you'll need to run a setup block first to get everything installed.

## ⚠️ GPU Constraints to know
- **Free Colab**: Gives you 1x T4 GPU (15GB VRAM). This is perfect for running `llama3.1:8b`. It does *not* have enough memory for `gemma2:27b`.
- **Colab Pro**: Gives you A100 or L4 GPUs, which have plenty of VRAM (24GB-40GB) to easily run massive models like `gemma2:27b` or `qwen2.5:32b`.

---

## 1. The "Nuke from Orbit" Setup Cell
Open a new Google Colab notebook, make sure you are using a GPU (`Runtime` -> `Change runtime type` -> `T4 GPU`), and paste this entire block into the very first cell. 

This will automatically install everything, start the server, clone your repository, and set up the `.env` file in one shot:

```python
# 1. Install required system packages
!apt-get update && apt-get install -y zstd

# 2. Install Ollama
!curl -fsSL https://ollama.com/install.sh | sh

# 3. Start Ollama silently in the background
import subprocess
import time
import os

print("Starting Ollama server...")
subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3)

# 4. Pull the model llama3.1:8b
print("Downloading model...")
!ollama pull gemma2:27b

# 5. Clone the repository and install dependencies
# We remove any existing folder first just in case you run this cell twice
!rm -rf PsyParse
!git clone https://github.com/manishsn7340/PsyParse.git
%cd PsyParse
!pip install -r requirements.txt

# 6. Create the .env file
with open('.env', 'w') as f:
    f.write('DEEPSEEK_API_KEY=ollama\n')
    f.write('DEEPSEEK_BASE_URL=http://localhost:11434/v1\n')
    f.write('DEEPSEEK_MODEL=gemma2:27b\n')

print("Setup Complete! Ready for evaluation.")
```

---

## 2. Run the Evaluation
Once that first cell finishes and prints "Setup Complete!", create a second code cell below it and run your evaluation:

```python
!PYTHONPATH=. python evaluation/run_eval.py 6
```

---


**Downloading results**: When it finishes, click the little **Folder icon** on the far left sidebar of Colab. Open `PsyParse` -> `results`, hover over `eval_results.json`, click the three dots, and select **Download**.

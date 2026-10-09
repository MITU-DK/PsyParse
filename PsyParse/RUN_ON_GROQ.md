# Running PsyParse with Groq Cloud (API) on Kaggle / Colab

This guide explains how to run PsyParse using the Groq API instead of a local model like Ollama. This approach is extremely fast and doesn't require downloading heavy models or installing Ollama, meaning it will run perfectly on the free tier of Kaggle or Colab.

## 1. The Setup Cell
Open a new notebook and paste this entire block into the very first cell. 

**Important:** Before running this cell, you must replace `YOUR_GROQ_API_KEY_HERE` with your actual Groq API key!

```python
import os
import shutil
import json
import urllib.request

# 1. Force root directory and wipe everything to fix the nested folder bug
os.chdir('/kaggle/working')
for item in os.listdir('.'):
    if os.path.isdir(item): shutil.rmtree(item)
    else: os.remove(item)

# 2. Clone the repository and change directory
!git clone https://github.com/manishsn7340/PsyParse.git
%cd /kaggle/working/PsyParse/PsyParse

# 3. Install dependencies
!pip install -r requirements.txt faiss-cpu sentence-transformers rank_bm25

# 4. Setup the .env file with Auto-Discovery for Groq Models
API_KEY = "YOUR_GROQ_API_KEY_HERE"

print("\nDiscovering available Groq models...")
try:
    req = urllib.request.Request('https://api.groq.com/openai/v1/models', headers={'Authorization': f'Bearer {API_KEY}'})
    resp = urllib.request.urlopen(req)
    models = [m['id'] for m in json.loads(resp.read().decode())['data']]
    
    # Priority: llama 8b, then any llama, then anything
    preferred = [m for m in models if 'llama' in m.lower() and '8b' in m.lower() and 'tool' not in m.lower()]
    if not preferred: preferred = [m for m in models if 'llama' in m.lower()]
    if not preferred: preferred = models
        
    best_model = preferred[0]
    print(f"✅ Success! Auto-selected model: {best_model}")
    
    with open('.env', 'w') as f:
        f.write(f'DEEPSEEK_API_KEY={API_KEY}\n')
        f.write('DEEPSEEK_BASE_URL=https://api.groq.com/openai/v1\n')
        f.write(f'DEEPSEEK_MODEL={best_model}\n')
        
    print("Groq API Setup Complete! Ready for execution.")
except Exception as e:
    print(f"❌ Error fetching models. Check your API key. Error: {e}")
```

## 2. Train the Model (Run Grid Search)
Since you are using Groq, API calls are nearly instantaneous. You can run the grid search hyperparameter tuning very quickly. Create a new cell and run:

```python
# For Kaggle use /kaggle/working, for Colab use /content
%cd /kaggle/working/PsyParse/PsyParse
!PYTHONPATH=. python evaluation/grid_search.py
```

## 3. Run the Evaluation
After finding your optimal weights, run the full pipeline evaluation:

```python
!PYTHONPATH=. python evaluation/run_eval.py 1
```

---
**Downloading results**: When it finishes, click the little **Folder icon** on the far left sidebar of Kaggle/Colab. Navigate to `results`, hover over `eval_results.json` or `summary_table.md`, click the three dots, and select **Download**.

# Running PsyParse with Groq Cloud (API) on Kaggle / Colab

This guide explains how to run PsyParse using the Groq API instead of a local model like Ollama. This approach is extremely fast and doesn't require downloading heavy models or installing Ollama, meaning it will run perfectly on the free tier of Kaggle or Colab.

## 1. The Setup Cell
Open a new notebook and paste this entire block into the very first cell. 

**Important:** Before running this cell, you must replace `YOUR_GROQ_API_KEY_HERE` with your actual Groq API key!

```python
import os

# Clone the repository
!rm -rf PsyParse
!git clone https://github.com/manishsn7340/PsyParse.git

# Install dependencies
!pip install -r PsyParse/PsyParse/requirements.txt faiss-cpu sentence-transformers rank_bm25

# Setup the .env file for Groq Cloud
with open('PsyParse/PsyParse/.env', 'w') as f:
    f.write('DEEPSEEK_API_KEY=YOUR_GROQ_API_KEY_HERE\n')
    f.write('DEEPSEEK_BASE_URL=https://api.groq.com/openai/v1\n')
    f.write('DEEPSEEK_MODEL=llama-3.1-8b-instant\n')

print("Groq API Setup Complete! Ready for execution.")
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

# Running PsyParse Locally

Follow these steps to run the evaluation pipeline entirely on your local machine using Ollama.

## 1. Prerequisites
1. **Install Python** (3.9 or higher).
2. **Install Ollama**: Download it from [ollama.com](https://ollama.com/) and install it.

## 2. Setup the Project
Open your terminal and run these commands to clone the code and install dependencies:

```bash
git clone https://github.com/manishsn7340/PsyParse.git
cd PsyParse

# Create and activate a virtual environment (Recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required Python libraries
pip install -r requirements.txt
```

## 3. Pull the Local Model
Make sure the Ollama app is running on your computer, then download the model. 

*Note: If you have a high-end setup (24GB VRAM like an RTX 3090/4090 or a Mac with 32GB+ Unified Memory), the Top Pick is **qwen2.5:32b**. It rivals 70B models in structured reasoning, follows JSON schemas reliably for the Evaluator, and generates diverse therapeutic responses. Otherwise, you can use **qwen2.5:14b**.*

```bash
ollama pull qwen2.5:32b
ollama pull qwen2.5:14b
```

## 4. Configure Environment Variables
Create a file named `.env` in the `PsyParse` folder and add these exactly 3 lines to it:

```env
DEEPSEEK_API_KEY=ollama
DEEPSEEK_BASE_URL=http://localhost:11434/v1
DEEPSEEK_MODEL=qwen2.5:14b
```
*(If you pulled a different model, make sure to change `DEEPSEEK_MODEL` to match, e.g., `qwen2.5:32b`).*

## 5. Run the Data Preparation
Before running the evaluation, you must prepare the therapeutic vector database:

```bash
PYTHONPATH=. python scripts/data_prep.py
```
This will build the `therapy_vectors.index` needed for the RAG component.

## 6. Run the Evaluation
With everything set up, run the pipeline (e.g., for 6 scenarios):

```bash
PYTHONPATH=. python evaluation/run_eval.py 6
```

Your final results will be saved to `results/eval_results.json` and a summary markdown report will be generated in `results/eval_summary.md`!

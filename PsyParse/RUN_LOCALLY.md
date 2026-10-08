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

*Note: `llama3.1:8b` requires about 6GB of RAM. If you have a powerful Mac (M-series with 32GB+ RAM) or a strong gaming GPU, you can pull `gemma2:27b` instead for better results.*

```bash
ollama pull llama3.1:8b
```

## 4. Configure Environment Variables
Create a file named `.env` in the root of the `PsyParse` folder and add these exactly 3 lines to it:

```env
DEEPSEEK_API_KEY=ollama
DEEPSEEK_BASE_URL=http://localhost:11434/v1
DEEPSEEK_MODEL=llama3.1:8b
```
*(If you pulled a different model, make sure to change `DEEPSEEK_MODEL` to match).*

## 5. Run the Evaluation
With everything set up and your virtual environment activated, run the pipeline:

```bash
PYTHONPATH=. python evaluation/run_eval.py 6
```

Your final results will be saved to `results/eval_results.json`!

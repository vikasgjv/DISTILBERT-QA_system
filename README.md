# NLP Projects: Question Answering & Pronoun Resolution

Two NLP projects built with HuggingFace Transformers — a **Question Answering system** powered by DistilBERT with a Flask web interface, and a **zero-shot Pronoun Resolution** system using BERT's masked language model.

---

## Projects

### 1. Question Answering System (Flask Web App)

An extractive QA system that answers questions from a given context paragraph using `distilbert-base-cased-distilled-squad`.

**Features:**
- Interactive web UI to enter a context and ask questions
- Displays the extracted answer with a confidence score and visual bar
- Built-in SQuAD dataset explorer (browse 1000 training samples)
- REST API endpoints for programmatic use

**Files:**
- `app.py` — Flask backend with QA inference and API routes
- `qa_project.py` — Standalone CLI version of the QA system
- `templates/index.html` — Frontend UI

---

### 2. Pronoun Resolution (Zero-Shot)

Resolves pronoun references to the correct antecedent using BERT's Masked Language Model — no fine-tuning required.

**How it works:**
- For each candidate antecedent, replaces the pronoun in the sentence with that candidate
- Masks each subword token of the candidate one at a time
- Sums the log-probabilities BERT assigns to those tokens (Pseudo-Log-Likelihood)
- The candidate with the higher score is predicted as the correct antecedent

**Evaluated on:** [Definite Pronoun Resolution (DPR)](https://huggingface.co/datasets/community-datasets/definite_pronoun_resolution) dataset

**File:** `pronoun_resolution.py`

---

## Setup

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the Web App

```bash
python app.py
```

Then open [http://localhost:5500](http://localhost:5500) in your browser.

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Web UI |
| `POST` | `/ask` | Get an answer for a question + context |
| `GET`  | `/dataset-sample?index=<n>` | Fetch a SQuAD sample by index (0–999) |

**Example `/ask` request:**
```json
POST /ask
{
  "question": "Who invented the telephone?",
  "context": "Alexander Graham Bell is credited with inventing the telephone in 1876."
}
```

**Response:**
```json
{
  "answer": "Alexander Graham Bell",
  "score": 0.8921
}
```

---

### Run the CLI QA System

```bash
python qa_project.py
```

Enter a context paragraph and a question when prompted. Type `exit` to quit.

---

### Run Pronoun Resolution

```bash
python pronoun_resolution.py
```

This will:
1. Evaluate the model on the DPR test set and print accuracy + classification report
2. Run a few custom demo predictions

---

## Models Used

| Model | Source | Task |
|-------|--------|------|
| `distilbert-base-cased-distilled-squad` | [HuggingFace](https://huggingface.co/distilbert-base-cased-distilled-squad) | Extractive QA |
| `bert-base-uncased` | [HuggingFace](https://huggingface.co/bert-base-uncased) | Pronoun Resolution (MLM) |

Models are downloaded automatically on first run via HuggingFace Hub.

---

## Datasets

| Dataset | Source | Used In |
|---------|--------|---------|
| SQuAD | [HuggingFace Datasets](https://huggingface.co/datasets/rajpurkar/squad) | QA system (first 1000 train samples) |
| Definite Pronoun Resolution (DPR) | [HuggingFace Datasets](https://huggingface.co/datasets/community-datasets/definite_pronoun_resolution) | Pronoun resolution evaluation |

---

## Tech Stack

- [Flask](https://flask.palletsprojects.com/) — Web framework
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — Pretrained models
- [HuggingFace Datasets](https://huggingface.co/docs/datasets) — Dataset loading
- [PyTorch](https://pytorch.org/) — Model inference
- [scikit-learn](https://scikit-learn.org/) — Evaluation metrics

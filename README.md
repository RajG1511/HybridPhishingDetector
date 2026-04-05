# Hybrid Phishing Detection Framework

A multi-layer hybrid system for detecting AI-generated and traditional phishing emails with full interpretability.

## Architecture

The detection pipeline follows a **cascading triage** approach across four layers:

1. **Layer 1 — Cryptographic & Protocol Authentication**: SPF, DKIM, ARC validation and header mismatch detection.
2. **Layer 2 — Lexical & URL Feature Engineering**: URL entropy, length, homoglyph detection, domain age, IP presence, subdomain depth.
3. **Layer 3 — Semantic Analysis & Super Learner Ensemble**: TF-IDF + DistilBERT embeddings fed into stacked ML models (RF, SVM, XGBoost → Logistic Regression meta-learner) with a parallel BiLSTM.
4. **Layer 4 — RAG-Driven Contextual Profiling** *(optional)*: Lightweight LLM + user history vector DB for grey-zone emails.

An **XAI module** (SHAP + LIME) provides human-readable explanations for every prediction.

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Run tests
pytest tests/ -v
```

## Usage

```bash
# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Send a POST request to `/analyze` with a `.eml` file or raw email text to receive a risk score (0–100) and XAI explanation.

## Project Structure

See [CLAUDE.md](CLAUDE.md) for the full directory structure and development roadmap.

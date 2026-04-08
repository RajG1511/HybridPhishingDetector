# Hybrid Phishing Detection Framework

A multi-layer hybrid system for detecting AI-generated and traditional phishing emails with full interpretability.

## Architecture

The detection pipeline follows a **cascading triage** approach across four layers:

1. **Layer 1 — Cryptographic & Protocol Authentication**: SPF, DKIM, ARC validation and header mismatch detection.
2. **Layer 2 — Lexical & URL Feature Engineering**: URL entropy, length, homoglyph detection, IP presence, subdomain depth (plus optional Domain Intel utility).
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

## Training the Metadata + URL Model

The Layer 1/2 classifier is trained on real `.eml` files from two sources:

- **EPVME** (~49K malicious emails) — High-fidelity phishing with real protocol vulnerabilities.
- **SpamAssassin** (~4K legitimate emails) — Used as the baseline legitimate corpus.

### 🚩 Strategic Note: The "Balancing Quirk" & Trust Baseline
Currently, the phishing dataset is ~12x larger than the legitimate set. To ensure a production-ready model:
- **Trust Baseline**: We use a **70% Modern Auth** ratio for Legitimate emails (allowing 30% legacy noise like missing DKIM). This prevents false positives on clean corporate mail.
- **Training Cap**: We employ **Random Class Downsampling**, capping the malicious training set at **~4,153 samples** to match the legitimate corpus.
- **The Gap**: We are currently leaving **~45,000 high-quality phishing samples** "on the table." 

**Roadmap**: Scale the legitimate corpus to 20k-30k samples to unlock the full 53k dataset capacity.

```bash
# 1. Download and extract datasets into data/raw/
python scripts/download_datasets.py

# 2. Extract features from .eml files (parallel, ~2 min)
python scripts/extract_eml_features.py --max-workers 6

# 3. Train and compare XGBoost vs CatBoost
python scripts/train_metadata_url_model.py --compare --selection-metric f1

# 4. Verify detection
python scripts/run_metadata_url_demo.py --sample phishing
python scripts/run_metadata_url_demo.py --sample ham
```

## Usage

```bash
# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Send a POST request to `/analyze` with a `.eml` file or raw email text to receive a risk score (0–100) and XAI explanation.

## Project Structure

See [CLAUDE.md](CLAUDE.md) for the full directory structure and development roadmap.

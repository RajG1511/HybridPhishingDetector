# CLAUDE.md — Phishing Detection Project Setup Instructions

## Project Overview

This project implements a **Hybrid Phishing Detection Framework** that combines cryptographic protocol authentication, lexical URL analysis, semantic NLP-based text classification, Explainable AI (XAI), and an optional RAG-driven contextual profiling layer. The goal is to detect AI-generated and traditional phishing emails with high accuracy and full interpretability.

The architecture follows a **cascading triage pipeline** with four layers:

1. **Layer 1 — Cryptographic & Protocol Authentication**: SPF, DKIM, ARC validation and header mismatch detection.
2. **Layer 2 — Lexical & URL Feature Engineering**: URL entropy, length, homoglyph detection, IP presence, subdomain depth.
3. **Layer 3 — Semantic Analysis & Super Learner Ensemble**: TF-IDF + DistilBERT embeddings fed into stacked ML models (RF, SVM, XGBoost → Logistic Regression meta-learner) with a parallel BiLSTM.
4. **Layer 4 — RAG-Driven Contextual Profiling** *(optional/advanced)*: Lightweight LLM + user history vector DB for grey-zone emails.

An **Explainable AI (XAI) module** using SHAP and LIME runs on top of the classification output to provide human-readable explanations.

---

## Directory Structure

When setting up the project, create the following directory tree:

```
phishing-detector/
├── CLAUDE.md                          # This file — project instructions
├── README.md                          # Project readme (generate a basic one)
├── requirements.txt                   # Python dependencies
├── .env.example                       # Template for environment variables
├── .gitignore                         # Standard Python + data gitignore
├── setup.py                           # Package setup (optional)
│
├── config/
│   ├── __init__.py
│   ├── settings.py                    # Global config: paths, thresholds, model params
│   └── logging_config.py             # Logging setup
│
├── data/
│   ├── raw/                           # Unprocessed downloads — DO NOT commit large files
│   │   ├── epvme/                     # EPVME malicious .eml dataset (~49K files)
│   │   │   ├── _repo/                # Cloned GitHub repo with zip archives
│   │   │   └── eml/                  # Extracted .eml files (flattened)
│   │   ├── spamassassin/             # SpamAssassin legitimate email corpus (~4K files)
│   │   │   └── eml/                  # Extracted email files (flattened)
│   │   ├── phishing_traditional/      # Legacy human-written phishing datasets
│   │   ├── phishing_ai/              # AI-generated phishing datasets
│   │   └── urls/                      # URL-specific datasets
│   ├── processed/                     # Cleaned, tokenized, vectorized data
│   │   ├── features/                  # Extracted feature CSVs
│   │   │   └── eml_training_features.csv  # Primary training table for Layer 1/2 model
│   │   └── embeddings/               # Cached DistilBERT / Word2Vec embeddings
│   ├── synthetic/                     # Procedurally generated phishing samples
│   ├── raw/kaggle/                    # Kaggle CSV datasets (Enron, CEAS, etc.)
│   └── splits/                        # train / val / test splits (saved as .csv or .pkl)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA on raw datasets
│   ├── 02_feature_engineering.ipynb   # Prototyping feature extraction
│   ├── 03_model_training.ipynb        # Training experiments and evaluation
│   ├── 04_explainability.ipynb        # SHAP / LIME analysis
│   └── 05_integration_demo.ipynb     # End-to-end pipeline demo
│
├── src/
│   ├── __init__.py
│   │
│   ├── layer1_protocol/              # Layer 1: Cryptographic & Protocol Auth
│   │   ├── __init__.py
│   │   ├── header_parser.py          # Parse From, Reply-To, Return-Path, Received headers
│   │   ├── spf_checker.py            # SPF record validation
│   │   ├── dkim_verifier.py          # DKIM signature verification via dkimpy
│   │   └── arc_validator.py          # ARC chain validation
│   │
│   ├── layer2_url/                   # Layer 2: Lexical & URL Feature Engineering
│   │   ├── __init__.py
│   │   ├── url_extractor.py          # Regex-based URL extraction from email body
│   │   ├── lexical_features.py       # Length, entropy, IP presence, homoglyphs, subdomain depth
│   │   └── domain_intel.py           # Domain intelligence utility (Preserved for future use)
│   │
│   ├── layer3_semantic/              # Layer 3: Semantic Analysis & Ensemble
│   │   ├── __init__.py
│   │   ├── preprocessor.py           # HTML stripping, tokenization, lemmatization, stopword removal
│   │   ├── vectorizer.py             # TF-IDF sparse vectors + DistilBERT dense embeddings
│   │   ├── ml_models.py              # RF, SVM, XGBoost, Logistic Regression base learners
│   │   ├── bilstm_model.py           # BiLSTM deep learning model (PyTorch)
│   │   ├── chargru_model.py          # Character-level GRU model (PyTorch)
│   │   └── ensemble.py               # Super Learner stacking meta-classifier
│   │
│   ├── layer4_rag/                   # Layer 4: RAG Contextual Profiling (optional)
│   │   ├── __init__.py
│   │   ├── vector_store.py           # User email history vector database
│   │   ├── rag_engine.py             # Retrieval + LLM reasoning for grey-zone emails
│   │   └── user_profile.py           # Baseline communication pattern modeling
│   │
│   ├── explainability/               # XAI Module
│   │   ├── __init__.py
│   │   ├── lime_explainer.py         # LIME local explanations for individual predictions
│   │   ├── shap_explainer.py         # SHAP global + local feature attributions
│   │   └── narrative_generator.py    # LLM-powered natural language explanation output
│   │
│   ├── pipeline/                     # End-to-end orchestration
│   │   ├── __init__.py
│   │   ├── email_ingester.py         # .eml file parsing and raw data extraction
│   │   ├── cascade_pipeline.py       # Full Layer 1 → 2 → 3 → 4 cascade logic
│   │   ├── metadata_url_model.py     # Learned Layer 1/2 model wrapper (XGBoost)
│   │   └── risk_scorer.py            # Final 0–100 risk score with hybrid ML+rules floor
│   │
│   └── data/
│       ├── __init__.py
│       └── csv_processor.py          # Unified mapping for Kaggle Enron/CEAS data
│
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py            # Dataset downloading and loading utilities
│       ├── smote_balancer.py         # SMOTE class imbalance handling
│       ├── metrics.py                # Accuracy, Precision, Recall, F1, AUC-ROC helpers
│       └── adversarial.py            # Adversarial example generation for training hardening
│
├── api/
│   ├── __init__.py
│   ├── main.py                       # FastAPI application entry point
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── analyze.py                # POST /analyze — accept .eml or raw text
│   │   └── health.py                 # GET /health — service health check
│   └── schemas.py                    # Pydantic request/response models
│
├── frontend/                         # Dashboard UI (or Chrome extension)
│   └── (placeholder — built in Phase 5)
│
├── models/                           # Saved/serialized trained models
│   ├── ml/                           # Pickled sklearn models
│   ├── dl/                           # PyTorch .pt checkpoints
│   └── vectorizers/                  # Saved TF-IDF / tokenizer artifacts
│
├── docker/
│   ├── Dockerfile                    # Backend container definition
│   ├── Dockerfile.frontend           # Frontend container (Phase 5)
│   └── docker-compose.yml            # Multi-container orchestration
│
└── tests/
    ├── __init__.py
    ├── test_layer1.py                # Unit tests for protocol authentication
    ├── test_layer2.py                # Unit tests for URL feature extraction
    ├── test_layer3.py                # Unit tests for semantic pipeline
    ├── test_pipeline.py              # Integration tests for full cascade
    └── conftest.py                   # Shared pytest fixtures
```

---

## Initial Setup Commands

Run the following to scaffold the project:

```bash
# 1. Create the project root
mkdir -p phishing-detector && cd phishing-detector

# 2. Create all directories
mkdir -p config
mkdir -p data/{raw/{ham,phishing_traditional,phishing_ai,urls},processed/{features,embeddings},synthetic,splits}
mkdir -p notebooks
mkdir -p src/{layer1_protocol,layer2_url,layer3_semantic,layer4_rag,explainability,pipeline,utils}
mkdir -p api/routes
mkdir -p frontend
mkdir -p models/{ml,dl,vectorizers}
mkdir -p docker
mkdir -p tests

# 3. Create all __init__.py files
touch config/__init__.py
touch src/__init__.py
touch src/layer1_protocol/__init__.py
touch src/layer2_url/__init__.py
touch src/layer3_semantic/__init__.py
touch src/layer4_rag/__init__.py
touch src/explainability/__init__.py
touch src/pipeline/__init__.py
touch src/utils/__init__.py
touch api/__init__.py
touch api/routes/__init__.py
touch tests/__init__.py

# 4. Create placeholder Python files with docstrings (see File Stubs section below)
```

---

## Dependencies (requirements.txt)

```
# === Core ===
python-dotenv>=1.0.0
pyyaml>=6.0

# === Data & ML ===
numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.5.0
xgboost>=2.1.0
imbalanced-learn>=0.12.0       # SMOTE

# === Deep Learning ===
torch>=2.3.0
torchvision>=0.18.0

# === NLP ===
nltk>=3.9.0
spacy>=3.7.0
transformers>=4.44.0           # DistilBERT, BERT
tokenizers>=0.19.0

# === Email Parsing & Protocol ===
dkimpy>=1.1.0
dnspython>=2.6.0
python-whois>=0.9.0

# === Explainability ===
lime>=0.2.0
shap>=0.45.0

# === API ===
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.8.0
python-multipart>=0.0.9        # File uploads

# === Visualization & Notebooks ===
matplotlib>=3.9.0
seaborn>=0.13.0
jupyter>=1.0.0
ipykernel>=6.29.0

# === Testing ===
pytest>=8.3.0
pytest-asyncio>=0.24.0
httpx>=0.27.0                  # For testing FastAPI

# === Utilities ===
tqdm>=4.66.0
requests>=2.32.0
beautifulsoup4>=4.12.0         # HTML stripping from email bodies

# === Optional: RAG Layer ===
# chromadb>=0.5.0              # Vector store for user profiles
# langchain>=0.2.0             # RAG orchestration
```

---

## Datasets

### Primary Training Data (Layer 1/2 — Metadata + URL Model)

These datasets contain raw `.eml` files with real email headers, enabling the model to learn SPF/DKIM/ARC authentication patterns. Downloaded automatically via `scripts/download_datasets.py`.

| Dataset | Target Dir | Count | Notes |
|---|---|---|---|
| **EPVME** (malicious) | `data/raw/epvme/` | ~49K `.eml` | Real header attacks. 10% are adversarially upgraded. [GitHub](https://github.com/sunknighteric/EPVME-Dataset) |
| **SpamAssassin & Kaggle** (legitimate) | `data/raw/kaggle/` | ~40K emails | Authentic corporate ham. **70% are augmented** with modern auth (30% left as legacy noise). CSVs natively mapped via `csv_processor.py`. |

### Supplementary Data (Layer 3 — Semantic / NLP)

| Dataset | Target Dir | Notes |
|---|---|---|
| Human-LLM Generated Phishing-Legitimate Emails | `data/raw/phishing_ai/` | 4,000 emails — WormGPT + ChatGPT generated. [Kaggle](https://www.kaggle.com/datasets/francescogreco97/human-llm-generated-phishing-legitimate-emails) |
| PhishFuzzer | `data/raw/phishing_ai/` | 23,100 LLM-generated variants with URL + attachment metadata. [GitHub](https://github.com/DataPhish/PhishFuzzer) |
| Phishing Email Dataset (Naser) | `data/raw/phishing_traditional/` | 135,894 samples. [Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) |
| Seven Phishing Email Datasets | `data/raw/phishing_traditional/` | 203,000 emails with full body text. [Figshare](https://figshare.com/articles/dataset/Seven_Phishing_Email_Datasets/25432108) |

### URL Datasets (→ `data/raw/urls/`)
| Dataset | Source | Notes |
|---|---|---|
| TRAP4Phish 2025 | [UNB](https://www.unb.ca/cic/datasets/trap4phish2025.html) | 1M+ malicious/benign URLs + HTML pages, PDFs, Excel docs |

---

## Key Configuration (`config/settings.py`)

This file should define:

```python
from pathlib import Path

# === Pipeline Augmentation ===
DEFAULT_AUGMENT_RATIO = 0.90
DEFAULT_COMPROMISED_RATIO = 0.10

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
METADATA_URL_MODELS_DIR = MODELS_DIR / "metadata_url"

# Raw .eml dataset directories
EPVME_DATA_DIR = RAW_DIR / "epvme" / "eml"
SPAMASSASSIN_DATA_DIR = RAW_DIR / "spamassassin" / "eml"

# Processed feature tables
EML_TRAINING_FEATURES_PATH = FEATURES_DIR / "eml_training_features.csv"

# === Layer 2 / Risk Scoring ===
LAYER1_MAX_RISK_POINTS = 30
LAYER2_MAX_RISK_POINTS = 25
LAYER3_MAX_RISK_POINTS = 45
METADATA_URL_MODEL_PATH = METADATA_URL_MODELS_DIR / "metadata_url_model.joblib"

# === Model Hyperparameters ===
TFIDF_MAX_FEATURES = 10_000
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"

# === Ensemble Thresholds ===
GREY_ZONE_LOW = 0.40    # Below this → benign
GREY_ZONE_HIGH = 0.75   # Above this → phishing
# Between LOW and HIGH → escalate to RAG layer

# === API ===
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_INFERENCE_LATENCY_MS = 200
```

---

## File Stubs

When creating each Python file, include a module-level docstring describing its purpose. Example:

**`src/layer1_protocol/header_parser.py`**
```python
"""
Layer 1 — Email Header Parser

Parses raw .eml files to extract and compare the From, Reply-To, and
Return-Path header fields. Flags mismatches as indicators of domain spoofing.

Dependencies: Python standard library `email` module.
"""
```

**`src/layer3_semantic/preprocessor.py`**
```python
"""
Layer 3 — NLP Text Preprocessor

Cleans raw email body text for downstream vectorization:
  - Strips HTML tags, JavaScript, CSS, and special characters
  - Tokenizes using NLTK or SpaCy
  - Removes stop words
  - Applies lemmatization

Input:  raw email body string
Output: list of cleaned tokens or a single cleaned string
"""
```

**`src/pipeline/cascade_pipeline.py`**
```python
"""
Cascade Pipeline — Full Detection Orchestration

Executes the four-layer detection cascade:
  1. Protocol authentication (Layer 1)
  2. URL feature analysis (Layer 2) — if links present
  3. Semantic ensemble classification (Layer 3)
  4. RAG contextual profiling (Layer 4) — only for grey-zone scores

Returns a final risk score (0–100) and an XAI explanation object.
"""
```

Apply this pattern to every file listed in the directory structure above.

---

## .gitignore

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
*.egg

# Virtual environments
venv/
.venv/
env/

# Data — large files should not be committed
data/raw/
data/processed/
data/synthetic/
data/splits/

# Models — serialized weights
models/

# Jupyter
.ipynb_checkpoints/
*.ipynb_metadata/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# Docker
docker/*.log

# OS
.DS_Store
Thumbs.db
```

---

## .env.example

```
# API keys for optional external services
VIRUSTOTAL_API_KEY=
CTI_PLATFORM_API_KEY=

# RAG layer (optional)
LLM_API_ENDPOINT=
LLM_MODEL_NAME=llama4-scout

# General
LOG_LEVEL=INFO
ENVIRONMENT=development

# Ratios
EML_AUGMENT_RATIO=0.70
EML_COMPROMISED_RATIO=0.10
```

---

## Phase-to-Directory Mapping

| Roadmap Phase | Days | Primary Directories |
|---|---|---|
| Phase 1: Data Acquisition | 1–10 | `data/raw/`, `data/synthetic/`, `src/utils/data_loader.py` |
| Phase 2: Feature Engineering | 11–20 | `src/layer1_protocol/`, `src/layer2_url/`, `src/layer3_semantic/preprocessor.py`, `src/layer3_semantic/vectorizer.py` |
| Phase 3: Model Training | 21–30 | `src/layer3_semantic/ml_models.py`, `src/layer3_semantic/bilstm_model.py`, `src/layer3_semantic/ensemble.py`, `src/utils/` |
| Phase 4: XAI & RAG | 31–40 | `src/explainability/`, `src/layer4_rag/` |
| Phase 5: Deployment & UI | 41–50 | `api/`, `frontend/`, `docker/` |

---

## Coding Conventions

- **Python version**: 3.11+
- **Type hints**: Use them everywhere — function signatures, return types, class attributes.
- **Docstrings**: Google-style docstrings on all public functions and classes.
- **Logging**: Use `logging` module via `config/logging_config.py` — never bare `print()`.
- **Testing**: Write unit tests alongside implementation. Target 80%+ coverage on `src/`.
- **Data handling**: Use `pathlib.Path` for all file paths. Never hardcode absolute paths.
- **Model serialization**: Save sklearn models with `joblib`. Save PyTorch models as `.pt` state dicts.

---

## First Task After Scaffolding

Once the directory structure and stub files are created:

1. Create a Python virtual environment: `python -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the SpaCy English model: `python -m spacy download en_core_web_sm`
4. Download NLTK data: `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"`
5. Verify the setup by running: `pytest tests/ -v`

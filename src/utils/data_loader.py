"""
Utilities — Dataset Loader

Loads all phishing/legitimate email datasets from data/raw/, normalizes their
label columns to a consistent three-class scheme:
    "legitimate"     — benign/ham emails
    "phishing_human" — human-written phishing / spam / fraud
    "phishing_ai"    — LLM-generated phishing

Merges into a single DataFrame with columns:
    source_dataset  — provenance tag
    original_label  — raw label from source file
    unified_label   — one of the three classes above
    raw_text        — original body text (before preprocessing)
    cleaned_text    — empty string; filled by preprocessor later

Public API:
    load_all_datasets() -> pd.DataFrame
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import RAW_DIR

logger = logging.getLogger(__name__)

# ── Output schema ──────────────────────────────────────────────────────────────
OUTPUT_COLUMNS = ["source_dataset", "original_label", "unified_label", "raw_text", "cleaned_text"]


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_record(source: str, original_label: str, unified_label: str, raw_text: str) -> dict:
    return {
        "source_dataset": source,
        "original_label": str(original_label),
        "unified_label": unified_label,
        "raw_text": str(raw_text) if raw_text else "",
        "cleaned_text": "",
    }


def _read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with fallback encodings."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {path} with any encoding")


# ══════════════════════════════════════════════════════════════════════════════
# Per-source loaders
# ══════════════════════════════════════════════════════════════════════════════

def _load_traditional_csv(
    path: Path,
    text_col: str,
    source_name: str,
    subject_col: Optional[str] = None,
    spam_is_human_phishing: bool = True,
) -> list[dict]:
    """Load a traditional phishing CSV (label 0=legit, 1=phishing/spam).

    Args:
        path: CSV file path.
        text_col: Column name for email body.
        source_name: Provenance tag.
        subject_col: Optional subject column to prepend to body.
        spam_is_human_phishing: Map label=1 to "phishing_human" (True) or "spam".
    """
    df = _read_csv_safe(path)
    records: list[dict] = []

    if text_col not in df.columns:
        logger.warning("%s missing column '%s'; skipping", path.name, text_col)
        return records

    for _, row in df.iterrows():
        raw = str(row.get(text_col, "") or "")
        if subject_col and subject_col in df.columns:
            subj = str(row.get(subject_col, "") or "")
            raw = f"{subj} {raw}".strip()

        label_raw = row.get("label", 0)
        try:
            label_int = int(label_raw)
        except (ValueError, TypeError):
            label_int = -1

        if label_int == 0:
            unified = "legitimate"
        elif label_int == 1:
            unified = "phishing_human" if spam_is_human_phishing else "spam"
        else:
            unified = "phishing_human"  # All-phishing files (Nazario, Nigerian_Fraud)

        if not raw.strip():
            continue
        records.append(_make_record(source_name, str(label_raw), unified, raw))

    logger.info("Loaded %d records from %s", len(records), source_name)
    return records


def _load_human_generated_ai() -> list[dict]:
    """Load phishing_ai/human-generated/ legit.csv and phishing.csv."""
    base = RAW_DIR / "phishing_ai" / "human-generated"
    records: list[dict] = []

    for fname, unified_label in [("legit.csv", "legitimate"), ("phishing.csv", "phishing_human")]:
        path = base / fname
        if not path.exists():
            logger.warning("Missing: %s", path)
            continue
        df = _read_csv_safe(path)
        text_col = "body" if "body" in df.columns else "text"
        for _, row in df.iterrows():
            raw = str(row.get(text_col, "") or "")
            subj = str(row.get("subject", "") or "")
            combined = f"{subj} {raw}".strip()
            if not combined:
                continue
            records.append(_make_record(
                "human_generated_ai",
                str(row.get("label", fname.split(".")[0])),
                unified_label,
                combined,
            ))

    logger.info("Loaded %d records from human-generated-ai", len(records))
    return records


def _load_llm_generated_ai() -> list[dict]:
    """Load phishing_ai/llm-generated/ legit.csv and phishing.csv.

    Note: label column in these files is unreliable — we infer class from
    the filename. The phishing.csv is partially malformed (multi-line rows
    parsed into wrong columns); we recover usable rows by dropping nulls
    and filtering by minimum text length.
    """
    base = RAW_DIR / "phishing_ai" / "llm-generated"
    records: list[dict] = []

    # legit.csv — label column says 1 but file is legitimate emails
    path = base / "legit.csv"
    if path.exists():
        df = _read_csv_safe(path)
        text_col = "text" if "text" in df.columns else df.columns[0]
        for _, row in df.iterrows():
            raw = str(row.get(text_col, "") or "")
            if len(raw.strip()) < 20:
                continue
            records.append(_make_record("llm_generated_ai", "legit_file", "legitimate", raw))

    # phishing.csv — partially malformed; recover rows with substantive text
    path = base / "phishing.csv"
    if path.exists():
        df = _read_csv_safe(path)
        text_col = "text" if "text" in df.columns else df.columns[0]
        df = df.dropna(subset=[text_col])
        # Filter out rows that are just fragments (too short to be a full email)
        df = df[df[text_col].str.len() >= 80]
        for _, row in df.iterrows():
            raw = str(row[text_col])
            records.append(_make_record("llm_generated_ai", "phishing_file", "phishing_ai", raw))

    logger.info("Loaded %d records from llm-generated-ai", len(records))
    return records


def _load_physhfuzzer() -> list[dict]:
    """Load PhishFuzzer JSON files (original seed + entity-rephrased).

    Type mapping:
        Phishing → phishing_human (original/human) | phishing_ai (LLM)
        Spam     → phishing_human (treat spam as phishing_human)
        Valid    → legitimate
    """
    base = RAW_DIR / "phishing_ai" / "PhishFuzzer"
    records: list[dict] = []

    files = [
        ("PhishFuzzer_emails_original_seed_v1.json", "PhishFuzzer_original"),
        ("PhishFuzzer_emails_entity_rephrased_v1.json", "PhishFuzzer_rephrased"),
    ]

    for fname, source_name in files:
        path = base / fname
        if not path.exists():
            logger.warning("Missing: %s", path)
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        created_by_vals = {str(r.get("Created by") or "").lower() for r in data}
        is_llm_source = "llm" in created_by_vals and "human" not in created_by_vals

        for r in data:
            body = str(r.get("Body", "") or "")
            subj = str(r.get("Subject", "") or "")
            raw = f"{subj} {body}".strip()
            if not raw or len(raw) < 10:
                continue

            email_type = str(r.get("Type", "")).strip()
            created_by = str(r.get("Created by") or "").lower()

            if email_type == "Valid":
                unified = "legitimate"
            elif email_type in ("Phishing", "Spam"):
                if "llm" in created_by or is_llm_source:
                    unified = "phishing_ai"
                else:
                    unified = "phishing_human"
            else:
                unified = "phishing_human"

            records.append(_make_record(source_name, email_type, unified, raw))

        logger.info("Loaded %d records from %s", len(records), source_name)

    return records


# ══════════════════════════════════════════════════════════════════════════════
# Main public loader
# ══════════════════════════════════════════════════════════════════════════════

def load_all_datasets(shuffle: bool = True, random_state: int = 42) -> pd.DataFrame:
    """Load and unify all available datasets from data/raw/.

    Skips empty directories silently.

    Returns:
        DataFrame with columns:
            source_dataset, original_label, unified_label, raw_text, cleaned_text
    """
    all_records: list[dict] = []

    # ── Traditional phishing datasets ─────────────────────────────────────────
    trad = RAW_DIR / "phishing_traditional"
    if trad.exists():
        # Enron: real corporate email (label 0=legit, 1=spam)
        p = trad / "Enron.csv"
        if p.exists():
            all_records.extend(_load_traditional_csv(p, "body", "Enron", subject_col="subject"))

        # Ling spam corpus
        p = trad / "Ling.csv"
        if p.exists():
            all_records.extend(_load_traditional_csv(p, "body", "Ling", subject_col="subject"))

        # Nazario: all phishing (label always 1)
        p = trad / "Nazario.csv"
        if p.exists():
            all_records.extend(_load_traditional_csv(p, "body", "Nazario", subject_col="subject"))

        # Nigerian fraud: all phishing (label always 1)
        p = trad / "Nigerian_Fraud.csv"
        if p.exists():
            all_records.extend(_load_traditional_csv(
                p, "body", "Nigerian_Fraud", subject_col="subject"
            ))

        # SpamAssassin
        p = trad / "SpamAssasin.csv"
        if p.exists():
            all_records.extend(_load_traditional_csv(
                p, "body", "SpamAssasin", subject_col="subject"
            ))

        # phishing_email.csv: pre-cleaned merged dataset (text_combined)
        # Included as a separate source; dedup on cleaned_text later.
        p = trad / "phishing_email.csv"
        if p.exists():
            all_records.extend(
                _load_traditional_csv(p, "text_combined", "phishing_email_combined")
            )

    # ── AI-generated datasets ─────────────────────────────────────────────────
    if (RAW_DIR / "phishing_ai").exists():
        all_records.extend(_load_human_generated_ai())
        all_records.extend(_load_llm_generated_ai())
        all_records.extend(_load_physhfuzzer())

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame(all_records, columns=OUTPUT_COLUMNS)

    # Drop rows with empty raw_text
    df = df[df["raw_text"].str.strip().str.len() > 0].copy()

    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    logger.info(
        "Total dataset: %d rows | Label dist: %s",
        len(df),
        df["unified_label"].value_counts().to_dict(),
    )
    return df

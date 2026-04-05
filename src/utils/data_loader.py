"""
Utilities — Dataset Loader

Provides functions for loading and normalizing the various phishing and
legitimate email datasets into a unified pandas DataFrame. Handles CSV,
JSON, and raw .eml directory formats.

Expected DataFrame schema:
    - text (str): Email body text (plain or preprocessed)
    - label (int): 0 = benign, 1 = phishing, 2 = spam (multiclass optional)
    - source (str): Dataset name for provenance tracking
"""

import logging
from pathlib import Path

import pandas as pd

from config.settings import RAW_DIR

logger = logging.getLogger(__name__)


def load_csv_dataset(path: Path, text_col: str, label_col: str, source_name: str = "") -> pd.DataFrame:
    """Load a CSV phishing dataset and normalize to standard schema.

    Args:
        path: Path to the CSV file.
        text_col: Column name containing email body text.
        label_col: Column name containing integer labels.
        source_name: Optional provenance tag.

    Returns:
        DataFrame with columns: text, label, source.
    """
    df = pd.read_csv(path, low_memory=False)
    df = df[[text_col, label_col]].dropna()
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df["source"] = source_name or path.stem
    logger.info("Loaded %d samples from %s", len(df), path)
    return df[["text", "label", "source"]]


def load_eml_directory(directory: Path, label: int, source_name: str = "") -> pd.DataFrame:
    """Load all .eml files from a directory as a dataset.

    Args:
        directory: Path containing .eml files.
        label: Integer label to assign to all samples (e.g. 0 for ham).
        source_name: Optional provenance tag.

    Returns:
        DataFrame with columns: text, label, source.
    """
    from src.pipeline.email_ingester import ingest_eml_file

    records: list[dict] = []
    for eml_path in sorted(directory.glob("*.eml")):
        try:
            parsed = ingest_eml_file(eml_path)
            body = parsed.plain_body or parsed.html_body
            records.append({"text": body, "label": label, "source": source_name or directory.name})
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", eml_path, exc)

    df = pd.DataFrame(records)
    logger.info("Loaded %d .eml files from %s", len(df), directory)
    return df


def combine_datasets(dfs: list[pd.DataFrame], shuffle: bool = True, random_state: int = 42) -> pd.DataFrame:
    """Concatenate and shuffle multiple DataFrames.

    Args:
        dfs: List of DataFrames with standard schema.
        shuffle: Whether to shuffle the combined dataset.
        random_state: Seed for reproducibility.

    Returns:
        Combined and optionally shuffled DataFrame.
    """
    combined = pd.concat(dfs, ignore_index=True)
    if shuffle:
        combined = combined.sample(frac=1, random_state=random_state).reset_index(drop=True)
    logger.info("Combined dataset: %d total samples", len(combined))
    return combined

"""Extract metadata + URL features from raw .eml files for model training.

Reads every .eml file from the EPVME (malicious) and SpamAssassin (legitimate)
directories, runs them through the existing Layer 1 and Layer 2 pipelines, and
produces a single training CSV with the exact feature columns that the
MetadataURLModel consumes at inference time.

Synthetic metadata augmentation is applied to the SpamAssassin (legitimate)
emails to simulate modern authentication headers:
  - 90 % receive  spf=pass, dkim=pass, arc=pass  (modern corporate mail)
  - 10 % keep their original (missing) headers   (personal / legacy servers)

Usage:
    python scripts/extract_eml_features.py                    # defaults
    python scripts/extract_eml_features.py --max-workers 8    # more parallelism
    python scripts/extract_eml_features.py --augment-ratio 0.85  # custom ratio
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR, DEFAULT_COMPROMISED_RATIO, DEFAULT_AUGMENT_RATIO
from src.pipeline.metadata_url_model import DEFAULT_FEATURE_NAMES
from src.data.csv_processor import process_kaggle_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Directories
EPVME_EML_DIR = DATA_DIR / "raw" / "epvme" / "eml"
SPAMASSASSIN_EML_DIR = DATA_DIR / "raw" / "spamassassin" / "eml"
KAGGLE_CSV_DIR = DATA_DIR / "raw" / "kaggle"
OUTPUT_CSV = DATA_DIR / "processed" / "features" / "eml_training_features.csv"

# Augmentation defaults
DEFAULT_AUGMENT_RATIO_VAL = DEFAULT_AUGMENT_RATIO
DEFAULT_COMPROMISED_RATIO_VAL = DEFAULT_COMPROMISED_RATIO
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_WORKERS = 4


# ---------------------------------------------------------------------------
# Feature extraction (runs in worker processes)
# ---------------------------------------------------------------------------

def _extract_features_from_eml(eml_path: str) -> dict[str, Any] | None:
    """Parse a single .eml file and return its feature vector.

    This function is designed to run in a subprocess via ProcessPoolExecutor.
    It imports pipeline modules locally to avoid pickling issues.

    Returns None if the file cannot be parsed.
    """
    try:
        # Late imports so each worker process initialises independently
        from src.pipeline.email_ingester import ingest_eml_file, ParsedEmail
        from src.layer2_url.url_extractor import extract_urls
        from src.layer2_url.lexical_features import (
            aggregate_lexical_features,
            extract_lexical_features,
        )

        path = Path(eml_path)
        parsed = ingest_eml_file(path)

        # --- Layer 1: Protocol / Metadata features ---
        layer1 = _extract_layer1_features(parsed)

        # --- Layer 2: URL / Lexical features ---
        layer2 = _extract_layer2_features(parsed)

        # --- Build feature dict matching DEFAULT_FEATURE_NAMES ---
        features = _build_raw_feature_dict(layer1, layer2)
        features["_filename"] = path.name
        return features

    except Exception:
        return None


def _extract_layer1_features(parsed: Any) -> dict[str, Any]:
    """Extract Layer 1 protocol metadata from a ParsedEmail.

    Falls back to raw header parsing if the full Layer 1 module is unavailable.
    """
    try:
        from src.layer1_protocol.metadata_features import extract_metadata_features
        mf = extract_metadata_features(parsed)
        output = mf.to_output_dict()
        return output
    except Exception:
        pass

    # Fallback: direct header inspection
    headers = parsed.headers or {}
    auth_results = headers.get("authentication-results", "") or ""

    spf = "none"
    if "spf=pass" in auth_results.lower():
        spf = "pass"
    elif "spf=fail" in auth_results.lower():
        spf = "fail"
    elif "spf=softfail" in auth_results.lower():
        spf = "softfail"

    dkim = "missing"
    if "dkim=pass" in auth_results.lower():
        dkim = "pass"
    elif "dkim=fail" in auth_results.lower():
        dkim = "fail"

    arc = "missing"
    if "arc=pass" in auth_results.lower():
        arc = "pass"
    elif "arc=fail" in auth_results.lower():
        arc = "fail"

    # Header mismatch: compare From vs Return-Path / Reply-To domains
    from_addr = (parsed.from_addr or "").lower()
    return_path = (parsed.return_path or "").lower()
    reply_to = (parsed.reply_to or "").lower()

    header_issues = []
    from_domain = _extract_domain(from_addr)

    if return_path and from_domain:
        rp_domain = _extract_domain(return_path)
        if rp_domain and rp_domain != from_domain:
            header_issues.append("Return-Path mismatch")

    if reply_to and from_domain:
        rt_domain = _extract_domain(reply_to)
        if rt_domain and rt_domain != from_domain:
            header_issues.append("Reply-To mismatch")

    # Check for multiple From addresses (EPVME attack type)
    from_raw = headers.get("from", "") or ""
    if "," in from_raw:
        header_issues.append("Multiple From addresses")

    metadata_flags = []
    if spf == "none":
        metadata_flags.append("Missing SPF")
    if dkim == "missing":
        metadata_flags.append("Missing DKIM")
    if arc == "missing":
        metadata_flags.append("Missing ARC")
    if not headers.get("message-id"):
        metadata_flags.append("Missing Message-ID")
    if not headers.get("received"):
        metadata_flags.append("Missing Received")

    # Protocol risk score (simple heuristic)
    risk = 0
    if spf != "pass":
        risk += 15
    if dkim != "pass":
        risk += 15
    if header_issues:
        risk += min(len(header_issues) * 10, 20)
    risk = min(risk, 100)

    return {
        "spf": spf,
        "dkim": dkim,
        "arc": arc,
        "header_mismatch": bool(header_issues),
        "header_issues": header_issues,
        "metadata_flags": metadata_flags,
        "protocol_risk_score": risk,
    }


def _extract_layer2_features(parsed: Any) -> dict[str, Any]:
    """Extract Layer 2 URL/lexical features from a ParsedEmail."""
    from src.layer2_url.url_extractor import extract_urls
    from src.layer2_url.lexical_features import (
        aggregate_lexical_features,
        extract_lexical_features,
    )

    body = parsed.html_body or parsed.plain_body or ""
    urls = extract_urls(body, is_html=bool(parsed.html_body))
    feature_summary = aggregate_lexical_features(urls)

    url_flags: list[str] = []
    for url in urls:
        feats = extract_lexical_features(url)
        if feats.get("has_ip"):
            url_flags.append("IP-based URL")
        if feats.get("has_homoglyph"):
            url_flags.append("Homoglyph URL")
        if feats.get("suspicious_tld"):
            url_flags.append("Suspicious TLD")
        if feats.get("uses_shortener"):
            url_flags.append("URL shortener")
        if feats.get("has_at_symbol"):
            url_flags.append("At symbol in URL")

    return {
        "url_count": len(urls),
        "urls": urls,
        "feature_summary": feature_summary,
        "url_flags": list(dict.fromkeys(url_flags)),
    }


def _build_raw_feature_dict(
    layer1: dict[str, Any],
    layer2: dict[str, Any],
) -> dict[str, float]:
    """Build the numeric feature vector from layer outputs."""
    summary = layer2.get("feature_summary") or {}

    features: dict[str, float] = {name: 0.0 for name in DEFAULT_FEATURE_NAMES}
    features.update({
        "protocol_risk_score": _bf(layer1.get("protocol_risk_score"), 100.0),
        "header_mismatch_count": float(max(
            len(layer1.get("header_issues", [])),
            1 if layer1.get("header_mismatch") else 0,
        )),
        "metadata_flag_count": float(len(layer1.get("metadata_flags", []))),
        "url_count": float(layer2.get("url_count", 0)),
        "url_flag_count": float(len(layer2.get("url_flags", []))),
        "url_length_mean": _bf(summary.get("url_length_mean"), 10000.0),
        "url_length_max": _bf(summary.get("url_length_max"), 10000.0),
        "url_entropy_mean": _bf(summary.get("url_entropy_mean"), 10.0),
        "url_entropy_max": _bf(summary.get("url_entropy_max"), 10.0),
        "has_ip_max": _bf(summary.get("has_ip_max"), 1.0),
        "has_homoglyph_max": _bf(summary.get("has_homoglyph_max"), 1.0),
        "suspicious_tld_max": _bf(summary.get("suspicious_tld_max"), 1.0),
        "uses_shortener_max": _bf(summary.get("uses_shortener_max"), 1.0),
        "has_at_symbol_max": _bf(summary.get("has_at_symbol_max"), 1.0),
        "subdomain_depth_mean": _bf(summary.get("subdomain_depth_mean"), 50.0),
        "subdomain_depth_max": _bf(summary.get("subdomain_depth_max"), 50.0),
        "path_depth_mean": _bf(summary.get("path_depth_mean"), 100.0),
        "path_depth_max": _bf(summary.get("path_depth_max"), 100.0),
        "num_special_chars_mean": _bf(summary.get("num_special_chars_mean"), 100.0),
        "num_special_chars_max": _bf(summary.get("num_special_chars_max"), 100.0),
        "num_url_params_mean": _bf(summary.get("num_url_params_mean"), 100.0),
        "num_url_params_max": _bf(summary.get("num_url_params_max"), 100.0),
        "url_digit_ratio_mean": _bf(summary.get("url_digit_ratio_mean"), 1.0),
        "url_digit_ratio_max": _bf(summary.get("url_digit_ratio_max"), 1.0),
    })

    # One-hot encode protocol statuses
    features.update(_onehot("spf", layer1.get("spf"), ("pass", "fail", "softfail", "none")))
    features.update(_onehot("dkim", layer1.get("dkim"), ("pass", "fail", "missing")))
    features.update(_onehot("arc", layer1.get("arc"), ("pass", "fail", "missing")))

    return features


def _extract_domain(addr: str) -> str:
    """Extract the domain part from an email address string."""
    if "@" in addr:
        # Handle formats like "Name <user@domain.com>" and bare "user@domain.com"
        clean = addr.split("<")[-1].rstrip(">").strip()
        parts = clean.split("@")
        if len(parts) == 2:
            return parts[1].strip().lower()
    return ""


def _bf(value: Any, upper: float) -> float:
    """Bounded float conversion."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(v, upper))


def _onehot(prefix: str, raw: Any, known: tuple[str, ...]) -> dict[str, float]:
    """One-hot encode a status string."""
    norm = str(raw or "unknown").strip().lower()
    result = {f"{prefix}_{v}": 0.0 for v in (*known, "unknown")}
    if norm not in known:
        norm = "unknown"
    result[f"{prefix}_{norm}"] = 1.0
    return result


# ---------------------------------------------------------------------------
# Synthetic metadata augmentation
# ---------------------------------------------------------------------------

def augment_legitimate_metadata(
    df: pd.DataFrame,
    augment_ratio: float = DEFAULT_AUGMENT_RATIO,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Apply synthetic metadata augmentation to legitimate email rows.

    For augment_ratio (default 90%) of the legitimate rows, override
    SPF/DKIM/ARC to pass and clear header mismatches. This simulates
    modern corporate email with properly configured authentication.

    The remaining rows keep their original extracted values, simulating
    personal or legacy servers without full authentication.
    """
    rng = random.Random(seed)
    legit_mask = df["label"] == 0
    legit_indices = df.index[legit_mask].tolist()

    n_augment = int(len(legit_indices) * augment_ratio)
    augmented_indices = set(rng.sample(legit_indices, n_augment))

    augmented = df.copy()

    for idx in augmented_indices:
        # Modern corporate email: everything passes
        augmented.at[idx, "spf_pass"] = 1.0
        augmented.at[idx, "spf_fail"] = 0.0
        augmented.at[idx, "spf_softfail"] = 0.0
        augmented.at[idx, "spf_none"] = 0.0
        augmented.at[idx, "spf_unknown"] = 0.0
        augmented.at[idx, "dkim_pass"] = 1.0
        augmented.at[idx, "dkim_fail"] = 0.0
        augmented.at[idx, "dkim_missing"] = 0.0
        augmented.at[idx, "dkim_unknown"] = 0.0
        augmented.at[idx, "arc_pass"] = 1.0
        augmented.at[idx, "arc_fail"] = 0.0
        augmented.at[idx, "arc_missing"] = 0.0
        augmented.at[idx, "arc_unknown"] = 0.0
        augmented.at[idx, "header_mismatch_count"] = 0.0
        augmented.at[idx, "metadata_flag_count"] = 0.0
        augmented.at[idx, "protocol_risk_score"] = 0.0

    logger.info(
        "Augmented %d / %d legitimate emails with modern auth headers (%.0f%% ratio)",
        n_augment,
        len(legit_indices),
        augment_ratio * 100,
    )
    return augmented


def augment_malicious_metadata(
    df: pd.DataFrame,
    compromised_ratio: float = DEFAULT_COMPROMISED_RATIO_VAL,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Apply adversarial data augmentation to malicious email rows.

    For compromised_ratio (default 50%) of the malicious rows, override
    SPF/DKIM/ARC to pass and clear header mismatches. This simulates
    Advanced Phishing / Compromised Inbox Phishing, preventing the ML model
    from heavily relying on simple authentication failure shortcuts.
    """
    rng = random.Random(seed)
    mal_mask = df["label"] == 1
    mal_indices = df.index[mal_mask].tolist()

    n_augment = int(len(mal_indices) * compromised_ratio)
    augmented_indices = set(rng.sample(mal_indices, n_augment))

    augmented = df.copy()

    for idx in augmented_indices:
        # Simulate perfect compromised corporate inbox
        augmented.at[idx, "spf_pass"] = 1.0
        augmented.at[idx, "spf_fail"] = 0.0
        augmented.at[idx, "spf_softfail"] = 0.0
        augmented.at[idx, "spf_none"] = 0.0
        augmented.at[idx, "spf_unknown"] = 0.0
        augmented.at[idx, "dkim_pass"] = 1.0
        augmented.at[idx, "dkim_fail"] = 0.0
        augmented.at[idx, "dkim_missing"] = 0.0
        augmented.at[idx, "dkim_unknown"] = 0.0
        augmented.at[idx, "arc_pass"] = 1.0
        augmented.at[idx, "arc_fail"] = 0.0
        augmented.at[idx, "arc_missing"] = 0.0
        augmented.at[idx, "arc_unknown"] = 0.0
        augmented.at[idx, "header_mismatch_count"] = 0.0
        augmented.at[idx, "metadata_flag_count"] = 0.0
        augmented.at[idx, "protocol_risk_score"] = 0.0

    logger.info(
        "Augmented %d / %d malicious emails to simulate clean Compromised Inboxes (%.0f%% ratio)",
        n_augment,
        len(mal_indices),
        compromised_ratio * 100,
    )
    return augmented


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract .eml features for training.")
    parser.add_argument(
        "--epvme-dir", type=Path, default=EPVME_EML_DIR,
        help="Directory containing EPVME .eml files.",
    )
    parser.add_argument(
        "--spamassassin-dir", type=Path, default=SPAMASSASSIN_EML_DIR,
        help="Directory containing SpamAssassin email files.",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_CSV,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--augment-ratio", type=float, default=DEFAULT_AUGMENT_RATIO,
        help="Fraction of legitimate emails augmented with modern headers (default: 0.90).",
    )
    parser.add_argument(
        "--compromise-ratio", type=float, default=DEFAULT_COMPROMISED_RATIO_VAL,
        help="Fraction of malicious emails augmented to simulate a compromised inbox (default: 0.50).",
    )
    parser.add_argument(
        "--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
        help="Number of parallel workers for feature extraction.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_RANDOM_SEED,
        help="Random seed for augmentation reproducibility.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit each dataset to N files (0 = no limit, useful for testing).",
    )
    return parser.parse_args()


def collect_eml_paths(directory: Path, limit: int = 0) -> list[Path]:
    """Collect .eml / email file paths from a directory."""
    if not directory.exists():
        logger.warning("Directory does not exist: %s", directory)
        return []

    # EPVME files end in .eml; SpamAssassin files may have no extension
    paths = sorted(directory.iterdir())
    paths = [p for p in paths if p.is_file()]
    if limit > 0:
        paths = paths[:limit]
    return paths


def extract_features_parallel(
    eml_paths: list[Path],
    label: int,
    source: str,
    max_workers: int,
) -> list[dict[str, Any]]:
    """Extract features from .eml files in parallel."""
    logger.info("Extracting features from %d %s files (workers=%d)…", len(eml_paths), source, max_workers)

    results: list[dict[str, Any]] = []
    failed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(_extract_features_from_eml, str(p)): p
            for p in eml_paths
        }

        done_count = 0
        for future in as_completed(future_map):
            done_count += 1
            if done_count % 2000 == 0 or done_count == len(eml_paths):
                logger.info("  [%s] Progress: %d / %d", source, done_count, len(eml_paths))

            result = future.result()
            if result is None:
                failed += 1
                continue

            result["label"] = label
            result["source"] = source
            results.append(result)

    logger.info(
        "  [%s] Extracted %d features (%d failed / %d total)",
        source, len(results), failed, len(eml_paths),
    )
    return results


def main() -> None:
    """Main feature extraction pipeline."""
    args = parse_args()

    # Collect file paths
    epvme_paths = collect_eml_paths(args.epvme_dir, limit=args.limit)
    spam_paths = collect_eml_paths(args.spamassassin_dir, limit=args.limit)

    if not epvme_paths and not spam_paths:
        logger.error(
            "No .eml files found. Run scripts/download_datasets.py first.\n"
            "  EPVME dir: %s\n  SpamAssassin dir: %s",
            args.epvme_dir, args.spamassassin_dir,
        )
        sys.exit(1)

    # Extract features in parallel
    all_rows: list[dict[str, Any]] = []

    if epvme_paths:
        malicious = extract_features_parallel(
            epvme_paths, label=1, source="epvme", max_workers=args.max_workers,
        )
        all_rows.extend(malicious)

    if spam_paths:
        legitimate = extract_features_parallel(
            spam_paths, label=0, source="spamassassin", max_workers=args.max_workers,
        )
        all_rows.extend(legitimate)

    # Process Kaggle CSV Sources
    if KAGGLE_CSV_DIR.exists():
        csv_files = list(KAGGLE_CSV_DIR.glob("*.csv"))
        if csv_files:
            logger.info("Processing %d Kaggle CSV sources...", len(csv_files))
            for csv_file in csv_files:
                csv_rows = process_kaggle_csv(csv_file)
                if csv_rows:
                    # Apply schema consistency (one-hots and bounding)
                    for row in csv_rows:
                        # Protocol one-hots
                        row.update(_onehot("spf", row.get("spf_status"), ("pass", "fail", "softfail", "none")))
                        row.update(_onehot("dkim", row.get("dkim_status"), ("pass", "fail", "missing")))
                        row.update(_onehot("arc", row.get("arc_status"), ("pass", "fail", "missing")))
                        
                        # Add missing defaults
                        for name in DEFAULT_FEATURE_NAMES:
                            if name not in row:
                                row[name] = 0.0
                        
                        row["source"] = f"kaggle_{csv_file.stem}"
                        row["_filename"] = f"{csv_file.stem}_row"
                        
                    all_rows.extend(csv_rows)

    if not all_rows:
        logger.error("No features were extracted successfully. Check logs for errors.")
        sys.exit(1)

    # Build DataFrame
    feature_cols = list(DEFAULT_FEATURE_NAMES) + ["label", "source", "_filename"]
    df = pd.DataFrame(all_rows, columns=feature_cols)

    # Fill any NaN values
    for col in DEFAULT_FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Apply synthetic metadata augmentation to legitimate emails
    df = augment_legitimate_metadata(df, augment_ratio=args.augment_ratio, seed=args.seed)

    # Apply adversarial compromised inbox augmentation to malicious emails
    df = augment_malicious_metadata(df, compromised_ratio=args.compromise_ratio, seed=args.seed)

    # Add row_number for compatibility with the training script
    df.insert(0, "row_number", range(len(df)))

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    # Summary statistics
    n_malicious = (df["label"] == 1).sum()
    n_legitimate = (df["label"] == 0).sum()
    logger.info("=" * 60)
    logger.info("Feature extraction complete!")
    logger.info("  Output: %s", args.output)
    logger.info("  Total rows: %d", len(df))
    logger.info("  Malicious (EPVME): %d", n_malicious)
    logger.info("  Legitimate (SpamAssassin): %d", n_legitimate)
    logger.info("  Class ratio: %.1f:1 (malicious:legitimate)", n_malicious / max(n_legitimate, 1))
    logger.info("  Feature columns: %d", len(DEFAULT_FEATURE_NAMES))

    # Quick distribution check
    logger.info("  --- Augmentation verification ---")
    legit = df[df["label"] == 0]
    logger.info("  Legitimate with spf_pass=1: %d / %d (%.1f%%)",
                (legit["spf_pass"] == 1.0).sum(), len(legit),
                (legit["spf_pass"] == 1.0).sum() / max(len(legit), 1) * 100)
    logger.info("  Legitimate with dkim_pass=1: %d / %d (%.1f%%)",
                (legit["dkim_pass"] == 1.0).sum(), len(legit),
                (legit["dkim_pass"] == 1.0).sum() / max(len(legit), 1) * 100)

    mal = df[df["label"] == 1]
    logger.info("  Malicious with header_mismatch>0: %d / %d (%.1f%%)",
                (mal["header_mismatch_count"] > 0).sum(), len(mal),
                (mal["header_mismatch_count"] > 0).sum() / max(len(mal), 1) * 100)
    logger.info("  Malicious upgraded to clean spf_pass=1 (Adversarial): %d / %d (%.1f%%)",
                (mal["spf_pass"] == 1.0).sum(), len(mal),
                (mal["spf_pass"] == 1.0).sum() / max(len(mal), 1) * 100)
    

    logger.info("=" * 60)


if __name__ == "__main__":
    main()

"""Download and extract the EPVME and SpamAssassin datasets for training.

This script fetches two complementary email corpora:

1. **EPVME** (Exploiting Protocol Vulnerabilities in Malicious Emails):
   ~49,136 raw .eml files containing real header-manipulation attacks
   (SPF/DMARC spoofing, XSS injection, subdomain attacks, CMS/MIME attacks).
   Source: https://github.com/sunknighteric/EPVME-Dataset

2. **SpamAssassin Public Corpus** (legitimate / ham emails):
   ~7,000 raw email files with authentic routing headers.
   Source: https://spamassassin.apache.org/old/publiccorpus/

Usage:
    python scripts/download_datasets.py            # download both
    python scripts/download_datasets.py --epvme     # EPVME only
    python scripts/download_datasets.py --spamassassin  # SpamAssassin only
    python scripts/download_datasets.py --skip-extract  # download only, no extraction
"""

from __future__ import annotations

import argparse
import io
import logging
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

EPVME_DIR = DATA_RAW / "epvme"
SPAMASSASSIN_DIR = DATA_RAW / "spamassassin"

EPVME_REPO_URL = "https://github.com/sunknighteric/EPVME-Dataset.git"
EPVME_ZIP_COUNT = 8

SPAMASSASSIN_BASE = "https://spamassassin.apache.org/old/publiccorpus/"
SPAMASSASSIN_FILES = [
    "20030228_easy_ham.tar.bz2",
    "20030228_easy_ham_2.tar.bz2",
    "20030228_hard_ham.tar.bz2",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download training datasets.")
    parser.add_argument("--epvme", action="store_true", help="Download EPVME only.")
    parser.add_argument("--spamassassin", action="store_true", help="Download SpamAssassin only.")
    parser.add_argument("--skip-extract", action="store_true", help="Download archives without extracting.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# EPVME
# ---------------------------------------------------------------------------

def download_epvme(*, extract: bool = True) -> Path:
    """Clone the EPVME-Dataset repository and extract all zip archives."""
    clone_dir = EPVME_DIR / "_repo"

    if clone_dir.exists():
        logger.info("EPVME repo already cloned at %s — skipping clone.", clone_dir)
    else:
        logger.info("Cloning EPVME-Dataset repository (≈170 MB)…")
        clone_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", EPVME_REPO_URL, str(clone_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Clone complete.")

    if not extract:
        return clone_dir

    eml_output = EPVME_DIR / "eml"
    eml_output.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(clone_dir.glob("EPVME_*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No EPVME_*.zip files found in {clone_dir}")

    logger.info("Found %d EPVME zip archives. Extracting…", len(zip_files))
    total_extracted = 0

    for zip_path in zip_files:
        logger.info("  Extracting %s…", zip_path.name)
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Each zip contains a numbered subdirectory (e.g. "1/", "2/")
            members = [m for m in zf.namelist() if m.endswith(".eml")]
            for member in members:
                # Flatten into our eml/ directory
                filename = Path(member).name
                target = eml_output / filename
                if target.exists():
                    continue  # skip duplicates
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                total_extracted += 1

    logger.info("EPVME extraction complete: %d .eml files in %s", total_extracted, eml_output)
    return eml_output


# ---------------------------------------------------------------------------
# SpamAssassin
# ---------------------------------------------------------------------------

def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
    """Simple progress reporter for urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        print(f"\r  Progress: {pct}% ({downloaded // 1024} KB / {total_size // 1024} KB)", end="", flush=True)


def download_spamassassin(*, extract: bool = True) -> Path:
    """Download and extract SpamAssassin ham corpora."""
    SPAMASSASSIN_DIR.mkdir(parents=True, exist_ok=True)
    eml_output = SPAMASSASSIN_DIR / "eml"
    eml_output.mkdir(parents=True, exist_ok=True)

    total_extracted = 0

    for filename in SPAMASSASSIN_FILES:
        archive_path = SPAMASSASSIN_DIR / filename
        url = f"{SPAMASSASSIN_BASE}{filename}"

        if archive_path.exists():
            logger.info("Archive %s already exists — skipping download.", filename)
        else:
            logger.info("Downloading %s…", url)
            try:
                urlretrieve(url, archive_path, reporthook=_reporthook)
                print()  # newline after progress bar
            except Exception as exc:
                logger.error("Failed to download %s: %s", url, exc)
                continue

        if not extract:
            continue

        logger.info("  Extracting %s…", filename)
        try:
            with tarfile.open(archive_path, "r:bz2") as tf:
                for member in tf.getmembers():
                    if member.isfile():
                        # Flatten: extract all files into eml/ with unique names
                        # Original structure is like easy_ham/00001.7c6aa2b9...
                        basename = Path(member.name).name
                        # Prefix with archive name to avoid collisions
                        prefix = filename.replace(".tar.bz2", "").replace("20030228_", "")
                        target = eml_output / f"{prefix}_{basename}"
                        if target.exists():
                            continue
                        with tf.extractfile(member) as src_file:
                            if src_file is not None:
                                target.write_bytes(src_file.read())
                                total_extracted += 1
        except Exception as exc:
            logger.error("Failed to extract %s: %s", filename, exc)
            continue

    logger.info("SpamAssassin extraction complete: %d email files in %s", total_extracted, eml_output)
    return eml_output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Download and extract datasets."""
    args = parse_args()
    download_both = not args.epvme and not args.spamassassin
    extract = not args.skip_extract

    if download_both or args.epvme:
        try:
            epvme_path = download_epvme(extract=extract)
            eml_count = len(list(epvme_path.glob("*.eml"))) if extract else 0
            logger.info("✓ EPVME ready: %d .eml files", eml_count)
        except Exception as exc:
            logger.error("✗ EPVME download failed: %s", exc)

    if download_both or args.spamassassin:
        try:
            spam_path = download_spamassassin(extract=extract)
            eml_count = len(list(spam_path.glob("*"))) if extract else 0
            logger.info("✓ SpamAssassin ready: %d email files", eml_count)
        except Exception as exc:
            logger.error("✗ SpamAssassin download failed: %s", exc)

    logger.info("Done. Run scripts/extract_eml_features.py next to build the training CSV.")


if __name__ == "__main__":
    main()

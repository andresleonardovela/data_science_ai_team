"""Data ingestion — downloads and validates the telecom churn dataset from Kaggle."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
from dotenv import load_dotenv

from tools.kaggle_scraper import KaggleScraperTool

load_dotenv()

DATASET_ID = "blastchar/telco-customer-churn"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def ingest() -> Path:
    """Download the Telco churn dataset and return path to the CSV file."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    scraper = KaggleScraperTool()
    result = scraper._run(dataset=DATASET_ID, output_dir=str(RAW_DIR))
    print(result)

    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {RAW_DIR} after download.")

    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Churn distribution:\n{df['Churn'].value_counts(normalize=True)}")

    return csv_path


if __name__ == "__main__":
    path = ingest()
    print(f"\nDataset path: {path}")

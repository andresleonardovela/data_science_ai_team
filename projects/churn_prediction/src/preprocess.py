"""Preprocessing — SMOTE class balancing and feature engineering for churn prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Columns to drop (non-predictive identifiers)
DROP_COLS = ["customerID"]

# Binary/ordinal columns to encode
BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]

# Multi-class nominal columns to one-hot encode
NOMINAL_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]


def preprocess(csv_path: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load, clean, encode, balance with SMOTE, and split the dataset.

    Returns:
        X_train, X_test, y_train, y_test (all numpy arrays)
    """
    if csv_path is None:
        candidates = list(RAW_DIR.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV in {RAW_DIR}. Run ingest.py first.")
        csv_path = candidates[0]

    df = pd.read_csv(csv_path)

    # Drop non-predictive columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Fix TotalCharges (occasionally has blank strings)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode binary columns
    le = LabelEncoder()
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # One-hot encode nominal columns
    df = pd.get_dummies(df, columns=[c for c in NOMINAL_COLS if c in df.columns], drop_first=True)

    # Split features / target
    X = df.drop(columns=["Churn"]).values.astype(np.float32)
    y = df["Churn"].values.astype(np.int32)

    # Scale numeric features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/test split before SMOTE (apply SMOTE only to training data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE to address class imbalance in training set only
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE — Training set: {X_train.shape}, class balance: {np.bincount(y_train)}")
    print(f"Test set: {X_test.shape}")

    # Persist processed splits
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    print(f"Processed data saved to {PROCESSED_DIR}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess()

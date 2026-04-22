"""Model training — Random Forest with precision/recall optimisation and MLflow tracking."""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models"

# Tuned for churn: high recall to catch at-risk customers, acceptable precision
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",  # Additional imbalance correction beyond SMOTE
    "random_state": 42,
    "n_jobs": -1,
}


def train(experiment_name: str = "churn_prediction") -> str:
    """Train a Random Forest model and log to MLflow.

    Returns:
        MLflow run ID for downstream tracking.
    """
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(RF_PARAMS)

        model = RandomForestClassifier(**RF_PARAMS)

        # Cross-validated recall on training data (prioritised metric)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring="recall").mean()
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1").mean()

        mlflow.log_metric("cv_recall", cv_recall)
        mlflow.log_metric("cv_f1", cv_f1)
        print(f"CV Recall: {cv_recall:.4f} | CV F1: {cv_f1:.4f}")

        # Final fit on full training set
        model.fit(X_train, y_train)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "random_forest.joblib"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(model_path))

        run_id = run.info.run_id
        print(f"Model saved: {model_path}")
        print(f"MLflow run ID: {run_id}")
        return run_id


if __name__ == "__main__":
    train()

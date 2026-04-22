"""Model evaluation — precision, recall, F1, AUC-ROC with QA validation hooks."""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models"

# QA thresholds — model must pass ALL of these to be approved for deployment
QA_THRESHOLDS = {
    "recall": 0.70,     # Must catch 70%+ of churners
    "precision": 0.55,  # Must be sufficiently precise to avoid false alarms
    "f1": 0.60,
    "roc_auc": 0.75,
}


def evaluate(model_path: Path | None = None) -> dict[str, float]:
    """Evaluate the trained model and validate against QA thresholds.

    Returns:
        Dictionary of metric name → value.
    Raises:
        ValueError if any QA threshold is not met.
    """
    if model_path is None:
        model_path = MODEL_DIR / "random_forest.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print("\n=== Evaluation Report ===")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nROC-AUC: {metrics['roc_auc']:.4f}")

    # Log to MLflow if an active run exists
    try:
        with mlflow.start_run(run_name="evaluation", nested=True):
            for name, value in metrics.items():
                mlflow.log_metric(f"test_{name}", value)
    except Exception:
        pass  # MLflow logging is optional during standalone evaluation

    # QA gate — validate against thresholds
    failures = {
        k: f"{v:.4f} < threshold {QA_THRESHOLDS[k]}"
        for k, v in metrics.items()
        if v < QA_THRESHOLDS[k]
    }
    if failures:
        msg = "\n".join(f"  {k}: {reason}" for k, reason in failures.items())
        raise ValueError(f"QA FAILED — model did not meet thresholds:\n{msg}")

    print("\nQA PASSED — all metric thresholds met.")
    return metrics


if __name__ == "__main__":
    evaluate()

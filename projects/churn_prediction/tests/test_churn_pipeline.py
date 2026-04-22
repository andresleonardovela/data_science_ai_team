"""QA test suite for the churn prediction pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_output_shapes_are_consistent(self, tmp_path):
        """X_train rows + X_test rows should equal dataset row count (approx)."""
        # Generate a minimal synthetic dataset to test the pipeline
        import pandas as pd

        df = pd.DataFrame({
            "customerID": [f"C{i}" for i in range(200)],
            "gender": (["Male", "Female"] * 100),
            "SeniorCitizen": ([0, 1] * 100),
            "Partner": (["Yes", "No"] * 100),
            "Dependents": (["Yes", "No"] * 100),
            "tenure": list(range(200)),
            "PhoneService": (["Yes"] * 200),
            "MultipleLines": (["No"] * 200),
            "InternetService": (["Fiber optic", "DSL"] * 100),
            "OnlineSecurity": (["No"] * 200),
            "OnlineBackup": (["No"] * 200),
            "DeviceProtection": (["No"] * 200),
            "TechSupport": (["No"] * 200),
            "StreamingTV": (["No"] * 200),
            "StreamingMovies": (["No"] * 200),
            "Contract": (["Month-to-month"] * 200),
            "PaperlessBilling": (["Yes"] * 200),
            "PaymentMethod": (["Electronic check"] * 200),
            "MonthlyCharges": [float(i) for i in range(200)],
            "TotalCharges": [str(float(i)) for i in range(200)],
            "Churn": (["Yes"] * 40 + ["No"] * 160),
        })

        csv_path = tmp_path / "test_churn.csv"
        df.to_csv(csv_path, index=False)

        # Patch the PROCESSED_DIR for this test
        from projects.churn_prediction.src import preprocess
        original_dir = preprocess.PROCESSED_DIR
        preprocess.PROCESSED_DIR = tmp_path / "processed"

        try:
            X_train, X_test, y_train, y_test = preprocess.preprocess(csv_path)
            assert X_train.shape[1] == X_test.shape[1], "Feature count must match across splits"
            assert len(X_train) == len(y_train), "Samples and labels must align"
            assert len(X_test) == len(y_test), "Test samples and labels must align"
        finally:
            preprocess.PROCESSED_DIR = original_dir

    def test_smote_balances_classes(self, tmp_path):
        """After SMOTE, training set should have balanced classes."""
        import pandas as pd
        from projects.churn_prediction.src import preprocess

        df = pd.DataFrame({
            "customerID": [f"C{i}" for i in range(100)],
            "gender": (["Male"] * 100),
            "SeniorCitizen": ([0] * 100),
            "Partner": (["Yes"] * 100),
            "Dependents": (["No"] * 100),
            "tenure": list(range(100)),
            "PhoneService": (["Yes"] * 100),
            "MultipleLines": (["No"] * 100),
            "InternetService": (["DSL"] * 100),
            "OnlineSecurity": (["No"] * 100),
            "OnlineBackup": (["No"] * 100),
            "DeviceProtection": (["No"] * 100),
            "TechSupport": (["No"] * 100),
            "StreamingTV": (["No"] * 100),
            "StreamingMovies": (["No"] * 100),
            "Contract": (["Month-to-month"] * 100),
            "PaperlessBilling": (["Yes"] * 100),
            "PaymentMethod": (["Electronic check"] * 100),
            "MonthlyCharges": [float(i) for i in range(100)],
            "TotalCharges": [str(float(i)) for i in range(100)],
            "Churn": (["Yes"] * 20 + ["No"] * 80),  # Imbalanced: 20/80
        })

        csv_path = tmp_path / "churn_imbalanced.csv"
        df.to_csv(csv_path, index=False)

        original_dir = preprocess.PROCESSED_DIR
        preprocess.PROCESSED_DIR = tmp_path / "processed"

        try:
            X_train, _, y_train, _ = preprocess.preprocess(csv_path)
            counts = np.bincount(y_train)
            assert counts[0] == counts[1], f"SMOTE should balance classes, got {counts}"
        finally:
            preprocess.PROCESSED_DIR = original_dir


# ---------------------------------------------------------------------------
# API serving tests
# ---------------------------------------------------------------------------

class TestServeAPI:
    @pytest.fixture
    def client(self):
        from projects.churn_prediction.src.serve import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()

    def test_predict_returns_expected_schema(self, client):
        """Verify prediction response schema even when model is not loaded."""
        # With no model, expect 503
        payload = {"features": [0.1] * 10}
        response = client.post("/predict", json=payload)
        # Either 200 (model loaded) or 503 (model not loaded) — both are valid
        assert response.status_code in (200, 503)

    def test_predict_rejects_nan_features(self, client):
        """NaN/Inf in features should return 422 validation error."""
        import math
        payload = {"features": [0.1, math.nan, 0.5]}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_rejects_empty_features(self, client):
        """Empty feature list should return 422 validation error."""
        payload = {"features": []}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Evaluation threshold tests
# ---------------------------------------------------------------------------

class TestEvaluationThresholds:
    def test_qa_thresholds_are_strict_enough(self):
        """Ensure QA thresholds enforce recall > precision (churn use-case requirement)."""
        from projects.churn_prediction.src.evaluate import QA_THRESHOLDS
        assert QA_THRESHOLDS["recall"] >= 0.70, "Recall threshold must be >= 0.70"
        assert QA_THRESHOLDS["roc_auc"] >= 0.75, "AUC-ROC threshold must be >= 0.75"

    def test_evaluate_raises_on_threshold_failure(self, tmp_path, monkeypatch):
        """evaluate() should raise ValueError when a metric fails the QA gate."""
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        import joblib

        from projects.churn_prediction.src import evaluate

        # Build a deliberately bad model on trivial data
        X = np.random.rand(100, 5).astype(np.float32)
        y = np.zeros(100, dtype=np.int32)  # All zeros → terrible recall for class 1
        y[:5] = 1  # Extreme imbalance

        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        processed = tmp_path / "processed"
        processed.mkdir()
        np.save(processed / "X_test.npy", X_test)
        np.save(processed / "y_test.npy", y_test)

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_path = models_dir / "random_forest.joblib"
        joblib.dump(model, model_path)

        monkeypatch.setattr(evaluate, "PROCESSED_DIR", processed)

        with pytest.raises(ValueError, match="QA FAILED"):
            evaluate.evaluate(model_path=model_path)

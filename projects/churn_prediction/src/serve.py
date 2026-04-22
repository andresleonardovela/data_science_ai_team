"""FastAPI model serving endpoint for churn prediction.

Run:
    uvicorn projects.churn_prediction.src.serve:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

MODEL_PATH = Path(__file__).parent.parent / "models" / "random_forest.joblib"

_model: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    else:
        # Allow startup without model for testing purposes
        _model = None
    yield
    _model = None


app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability using a Random Forest model.",
    version="1.0.0",
    lifespan=lifespan,
)


class ChurnFeatures(BaseModel):
    """Input features for churn prediction.

    Feature vector must match the preprocessing pipeline output.
    Send the scaled, encoded feature array as a flat list.
    """
    features: list[float] = Field(
        description="Preprocessed feature vector (output of preprocess.py).",
        min_length=1,
    )

    @field_validator("features")
    @classmethod
    def no_nan_or_inf(cls, v: list[float]) -> list[float]:
        if any(not np.isfinite(x) for x in v):
            raise ValueError("Feature vector must not contain NaN or Inf values.")
        return v


class ChurnPrediction(BaseModel):
    churn: bool
    churn_probability: float
    risk_level: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=ChurnPrediction)
def predict(payload: ChurnFeatures) -> ChurnPrediction:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    X = np.array(payload.features, dtype=np.float32).reshape(1, -1)
    churn_prob = float(_model.predict_proba(X)[0, 1])
    churn = churn_prob >= 0.5

    if churn_prob >= 0.75:
        risk = "HIGH"
    elif churn_prob >= 0.5:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return ChurnPrediction(
        churn=churn,
        churn_probability=round(churn_prob, 4),
        risk_level=risk,
    )

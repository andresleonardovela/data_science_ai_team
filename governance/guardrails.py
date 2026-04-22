"""watsonx.governance guardrails — bias detection and model drift monitoring hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.metrics import recall_score


@dataclass
class DriftReport:
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    baseline_recall: float = 0.0
    current_recall: float = 0.0
    drift_detected: bool = False
    drift_delta: float = 0.0
    recommendation: str = ""


@dataclass
class BiasReport:
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    group_metrics: dict[str, float] = field(default_factory=dict)
    max_disparity: float = 0.0
    bias_detected: bool = False
    threshold: float = 0.10
    recommendation: str = ""


class GovernanceMonitor:
    """Monitors trained models for drift and protected-group bias.

    Designed to be called post-deployment with a rolling window of predictions.
    Integrates with watsonx.governance Lite when credentials are configured.
    """

    DRIFT_THRESHOLD = 0.05   # Alert if recall drops > 5% from baseline
    BIAS_THRESHOLD = 0.10    # Alert if recall disparity > 10% across groups

    def __init__(self, baseline_recall: float) -> None:
        self.baseline_recall = baseline_recall

    def check_drift(self, y_true: np.ndarray, y_pred: np.ndarray) -> DriftReport:
        """Compare current recall against baseline to detect model drift."""
        current_recall = recall_score(y_true, y_pred, zero_division=0)
        delta = self.baseline_recall - current_recall
        drifted = delta > self.DRIFT_THRESHOLD

        report = DriftReport(
            baseline_recall=self.baseline_recall,
            current_recall=current_recall,
            drift_detected=drifted,
            drift_delta=round(delta, 4),
            recommendation=(
                "Retrain the model with recent data." if drifted else "No action required."
            ),
        )
        if drifted:
            print(f"[GOVERNANCE ALERT] Model drift detected: recall dropped {delta:.4f}")
        return report

    def check_bias(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: dict[str, np.ndarray],
    ) -> BiasReport:
        """Check for recall disparity across protected groups (e.g., gender, age).

        Args:
            y_true: Ground truth labels.
            y_pred: Model predictions.
            groups: Dict mapping group name → boolean mask array.
        """
        group_recalls: dict[str, float] = {}
        for group_name, mask in groups.items():
            if mask.sum() == 0:
                continue
            group_recalls[group_name] = round(
                recall_score(y_true[mask], y_pred[mask], zero_division=0), 4
            )

        if len(group_recalls) < 2:
            return BiasReport(group_metrics=group_recalls)

        values = list(group_recalls.values())
        disparity = max(values) - min(values)
        biased = disparity > self.BIAS_THRESHOLD

        report = BiasReport(
            group_metrics=group_recalls,
            max_disparity=round(disparity, 4),
            bias_detected=biased,
            threshold=self.BIAS_THRESHOLD,
            recommendation=(
                f"Investigate and mitigate recall disparity ({disparity:.4f}) across groups."
                if biased else "No significant bias detected."
            ),
        )
        if biased:
            print(f"[GOVERNANCE ALERT] Bias detected: max recall disparity = {disparity:.4f}")
        return report

    def _send_to_watsonx_governance(self, report: Any) -> None:
        """Placeholder — send report to IBM watsonx.governance API.

        Replace with actual REST call when IBM OpenScale/Watson OpenScale
        credentials are available in environment variables.
        """
        raise NotImplementedError("watsonx.governance API integration not yet configured.")

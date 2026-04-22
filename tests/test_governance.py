"""Unit tests for governance modules."""

from __future__ import annotations

import numpy as np
import pytest

from governance.toxicity_filter import ToxicityFilter
from governance.guardrails import GovernanceMonitor


class TestToxicityFilter:
    def setup_method(self):
        self.f = ToxicityFilter()

    def test_clean_text_passes(self):
        result = self.f.filter("The model achieved 82% recall on the test set.")
        assert result.passed
        assert result.violations == []

    def test_email_is_redacted(self):
        result = self.f.filter("Contact user@example.com for details.")
        assert "[EMAIL REDACTED]" in result.sanitised
        assert "user@example.com" not in result.sanitised

    def test_toxic_content_is_blocked(self):
        result = self.f.filter("Instructions on how to bypass security controls.")
        assert not result.passed
        assert "[RESPONSE BLOCKED]" in result.sanitised

    def test_wrap_llm_call_filters_output(self):
        def mock_llm(prompt: str) -> str:
            return "Here is how to bypass security and steal credentials."

        wrapped = self.f.wrap_llm_call(mock_llm)
        output = wrapped("safe prompt")
        assert "[RESPONSE BLOCKED]" in output


class TestGovernanceMonitor:
    def test_no_drift_when_within_threshold(self):
        monitor = GovernanceMonitor(baseline_recall=0.80)
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 1])  # recall = 7/8 = 0.875
        report = monitor.check_drift(y_true, y_pred)
        assert not report.drift_detected

    def test_drift_detected_when_recall_drops(self):
        monitor = GovernanceMonitor(baseline_recall=0.90)
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0])  # recall = 2/5 = 0.40
        report = monitor.check_drift(y_true, y_pred)
        assert report.drift_detected

    def test_bias_detected_when_disparity_exceeds_threshold(self):
        monitor = GovernanceMonitor(baseline_recall=0.80)
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # group A: 1.0, group B: 0.0
        groups = {
            "group_A": np.array([True, True, True, True, False, False, False, False]),
            "group_B": np.array([False, False, False, False, True, True, True, True]),
        }
        report = monitor.check_bias(y_true, y_pred, groups)
        assert report.bias_detected
        assert report.max_disparity > 0.10

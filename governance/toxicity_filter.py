"""Toxicity filter — pre/post LLM call output sanitisation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class FilterResult:
    passed: bool
    original: str
    sanitised: str
    violations: list[str]


# Toxic/policy-violating patterns (case-insensitive)
# Extend this list based on your organisation's content policy.
_TOXIC_PATTERNS: list[tuple[str, str]] = [
    (r"\b(kill|murder|attack|harm|destroy)\s+(people|users|customers)\b", "Violence directive"),
    (r"\b(hate|despise)\s+(race|religion|gender|ethnicity)\b", "Hate speech"),
    (r"\b(phish|scam|defraud|steal)\b", "Fraud instruction"),
    (r"\b(bypass|circumvent|disable)\s+(security|firewall|auth)\b", "Security bypass"),
    (r"\b(dump|exfiltrate|leak)\s+(credentials|passwords|tokens)\b", "Credential exfiltration"),
]

# PII patterns — redact from outputs
_PII_PATTERNS: list[tuple[str, str, str]] = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL REDACTED]", "Email address"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]", "SSN"),
    (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD REDACTED]", "Credit card"),
    (r"\b\+?[1-9]\d{1,14}\b", "[PHONE REDACTED]", "Phone number"),
]


class ToxicityFilter:
    """Filters LLM inputs and outputs for toxic content and PII.

    Usage:
        f = ToxicityFilter()
        result = f.filter(llm_output)
        if not result.passed:
            raise ValueError(f"Policy violation: {result.violations}")
        safe_text = result.sanitised
    """

    def __init__(self, redact_pii: bool = True, block_on_toxicity: bool = True) -> None:
        self.redact_pii = redact_pii
        self.block_on_toxicity = block_on_toxicity

    def filter(self, text: str) -> FilterResult:
        """Filter a string for toxic content and PII.

        Returns a FilterResult. If block_on_toxicity=True and violations are
        found, the sanitised text is replaced with a safe refusal message.
        """
        violations: list[str] = []
        sanitised = text

        # Check for toxic patterns
        for pattern, label in _TOXIC_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(label)

        # Redact PII
        if self.redact_pii:
            for pattern, replacement, label in _PII_PATTERNS:
                if re.search(pattern, sanitised, re.IGNORECASE):
                    sanitised = re.sub(pattern, replacement, sanitised, flags=re.IGNORECASE)

        passed = len(violations) == 0

        if not passed and self.block_on_toxicity:
            sanitised = (
                "[RESPONSE BLOCKED] This output was flagged for policy violations: "
                + ", ".join(violations)
            )

        return FilterResult(
            passed=passed,
            original=text,
            sanitised=sanitised,
            violations=violations,
        )

    def wrap_llm_call(self, llm_fn: Callable[[str], str]) -> Callable[[str], str]:
        """Decorator that applies this filter to both the prompt and the response."""
        def wrapped(prompt: str) -> str:
            input_check = self.filter(prompt)
            if not input_check.passed:
                return input_check.sanitised
            raw_response = llm_fn(input_check.sanitised)
            output_check = self.filter(raw_response)
            return output_check.sanitised
        return wrapped

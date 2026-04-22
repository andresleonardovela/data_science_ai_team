"""IBM Guardium vulnerability scanner tool (stub + integration hook)."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# Patterns that indicate potential security issues in code/outputs
_RISK_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)(password|passwd|pwd)\s*=\s*['\"][^'\"]+['\"]", "Hardcoded credential detected"),
    (r"(?i)api[_-]?key\s*=\s*['\"][^'\"]+['\"]", "Hardcoded API key detected"),
    (r"(?i)secret\s*=\s*['\"][^'\"]+['\"]", "Hardcoded secret detected"),
    (r"(?i)eval\s*\(", "Use of eval() — potential code injection risk"),
    (r"(?i)exec\s*\(", "Use of exec() — potential code injection risk"),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "Hardcoded IP address detected"),
]


class GuardiumInput(BaseModel):
    content: str = Field(description="Code or text content to scan for vulnerabilities.")
    content_type: str = Field(
        default="code",
        description="Type of content: 'code' or 'text'.",
    )


class GuardiumScannerTool(BaseTool):
    """Scans content for OWASP-aligned security vulnerabilities.

    Currently performs local static analysis. Hook `_call_guardium_api()`
    to connect to IBM Guardium when credentials are available.
    """

    name: str = "guardium_scanner"
    description: str = (
        "Scan code or text for security vulnerabilities including hardcoded credentials, "
        "injection risks, and policy violations. Returns a structured risk report."
    )
    args_schema: type[BaseModel] = GuardiumInput

    def _run(self, content: str, content_type: str = "code") -> str:
        findings: list[str] = []
        for pattern, label in _RISK_PATTERNS:
            matches = re.findall(pattern, content)
            if matches:
                findings.append(f"[HIGH] {label} — {len(matches)} occurrence(s)")

        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]

        if not findings:
            return f"SCAN PASSED — No vulnerabilities found. Content hash: {content_hash}"

        report = "\n".join(findings)
        return (
            f"SCAN FAILED — {len(findings)} issue(s) found. Content hash: {content_hash}\n\n"
            f"{report}\n\nRecommendation: Remediate all HIGH findings before deployment."
        )

    def _call_guardium_api(self, content: str) -> dict[str, Any]:
        """Placeholder for IBM Guardium API integration.

        Replace with actual IBM Guardium REST API call when credentials
        and endpoint are configured in environment variables.
        """
        raise NotImplementedError("IBM Guardium API integration not yet configured.")


def build_guardium_tool() -> GuardiumScannerTool:
    return GuardiumScannerTool()

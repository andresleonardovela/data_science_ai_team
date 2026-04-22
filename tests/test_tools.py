"""Unit tests for tool wrappers."""

from __future__ import annotations

import pytest

from tools.guardium_scanner import GuardiumScannerTool
from tools.registry import ToolRegistry


class TestGuardiumScanner:
    def setup_method(self):
        self.scanner = GuardiumScannerTool()

    def test_clean_code_passes(self):
        code = "def add(a, b):\n    return a + b"
        result = self.scanner._run(code)
        assert "SCAN PASSED" in result

    def test_hardcoded_password_is_flagged(self):
        code = "password = 'super_secret_123'"
        result = self.scanner._run(code)
        assert "SCAN FAILED" in result
        assert "Hardcoded credential" in result

    def test_eval_usage_is_flagged(self):
        code = "result = eval(user_input)"
        result = self.scanner._run(code)
        assert "SCAN FAILED" in result
        assert "eval()" in result

    def test_api_key_is_flagged(self):
        code = 'api_key = "sk-abc123xyz"'
        result = self.scanner._run(code)
        assert "SCAN FAILED" in result


class TestToolRegistry:
    def test_has_returns_true_for_known_tool(self):
        r = ToolRegistry()
        assert r.has("kaggle_scraper")
        assert r.has("rag_tool")
        assert r.has("guardium_scanner")
        assert r.has("fastapi_deployer")

    def test_has_returns_false_for_unknown_tool(self):
        r = ToolRegistry()
        assert not r.has("nonexistent_tool")

    def test_available_tools_returns_list(self):
        tools = ToolRegistry.available_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 4

    def test_get_raises_for_unknown_tool(self):
        r = ToolRegistry()
        with pytest.raises(ValueError, match="Unknown tool"):
            r.get("nonexistent_tool")

"""Tool registry: auto-discovers and serves tool wrappers by name."""

from __future__ import annotations

from typing import Any


class ToolRegistry:
    """Central registry for all agent tool wrappers.

    Tools are registered by name and lazily imported on first access.
    Agents request tools by the names listed in their YAML config.
    """

    _catalog: dict[str, tuple[str, str]] = {
        "rag_tool": ("tools.rag_tool", "build_rag_tool"),
        "kaggle_scraper": ("tools.kaggle_scraper", "build_kaggle_tool"),
        "guardium_scanner": ("tools.guardium_scanner", "build_guardium_tool"),
        "fastapi_deployer": ("tools.fastapi_deployer", "build_deployer_tool"),
    }

    def __init__(self) -> None:
        self._instances: dict[str, Any] = {}

    def has(self, name: str) -> bool:
        return name in self._catalog

    def get(self, name: str) -> Any:
        """Return a cached tool instance by name."""
        if name not in self._catalog:
            raise ValueError(f"Unknown tool: {name!r}. Available: {list(self._catalog)}")
        if name not in self._instances:
            import importlib
            module_path, factory = self._catalog[name]
            module = importlib.import_module(module_path)
            self._instances[name] = getattr(module, factory)()
        return self._instances[name]

    def get_all(self) -> list[Any]:
        return [self.get(name) for name in self._catalog]

    @staticmethod
    def available_tools() -> list[str]:
        return list(ToolRegistry._catalog.keys())

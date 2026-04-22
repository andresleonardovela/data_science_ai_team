"""Agent registry: auto-discovers and loads agents by name."""

from __future__ import annotations

import importlib
from typing import Any

from agents.base_agent import BaseAgent


# Maps role_name → module path and class name
_AGENT_CATALOG: dict[str, tuple[str, str]] = {
    "product_owner": ("agents.product_owner", "ProductOwnerAgent"),
    "data_architect": ("agents.data_architect", "DataArchitectAgent"),
    "data_engineer": ("agents.data_engineer", "DataEngineerAgent"),
    "data_scientist": ("agents.data_scientist", "DataScientistAgent"),
    "ml_engineer": ("agents.ml_engineer", "MLEngineerAgent"),
    "backend_engineer": ("agents.backend_engineer", "BackendEngineerAgent"),
    "qa_tester": ("agents.qa_tester", "QATesterAgent"),
    "cybersecurity": ("agents.cybersecurity", "CybersecurityAgent"),
}


class AgentRegistry:
    """Registry that instantiates agent objects by role name.

    Usage:
        registry = AgentRegistry(llm=llm, tool_registry=tool_registry)
        agent = registry.get("data_scientist")
        crewai_agent = agent.build()
    """

    def __init__(self, llm: Any, tool_registry: Any = None) -> None:
        self.llm = llm
        self.tool_registry = tool_registry
        self._instances: dict[str, BaseAgent] = {}

    def get(self, role_name: str) -> BaseAgent:
        """Return a cached agent instance by role name."""
        if role_name not in _AGENT_CATALOG:
            available = ", ".join(_AGENT_CATALOG.keys())
            raise ValueError(f"Unknown agent role: {role_name!r}. Available: {available}")

        if role_name not in self._instances:
            module_path, class_name = _AGENT_CATALOG[role_name]
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            self._instances[role_name] = cls(llm=self.llm, tool_registry=self.tool_registry)

        return self._instances[role_name]

    def get_all(self) -> list[BaseAgent]:
        """Instantiate and return all registered agents."""
        return [self.get(name) for name in _AGENT_CATALOG]

    @staticmethod
    def available_roles() -> list[str]:
        return list(_AGENT_CATALOG.keys())

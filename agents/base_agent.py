"""Base agent abstraction for all AI team agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml
from crewai import Agent
from pydantic import BaseModel

from tools.registry import ToolRegistry


class AgentConfig(BaseModel):
    """Schema for YAML agent configuration files."""

    name: str
    role: str
    goal: str
    backstory: str
    tools: list[str] = []
    verbose: bool = True
    allow_delegation: bool = False
    max_iter: int = 10
    memory: bool = True


class BaseAgent(ABC):
    """Abstract base class for all AI team agents.

    Subclasses must implement `role_name` and may override `build()`.
    Agent identity is driven by a YAML config file under agents/config/.
    """

    CONFIG_DIR = Path(__file__).parent / "config"

    def __init__(self, llm: Any, tool_registry: ToolRegistry | None = None) -> None:
        self.llm = llm
        self.tool_registry = tool_registry or ToolRegistry()
        self._config = self._load_config()
        self._agent: Agent | None = None

    @property
    @abstractmethod
    def role_name(self) -> str:
        """Filename stem of the YAML config (e.g. 'data_scientist')."""

    def _load_config(self) -> AgentConfig:
        config_path = self.CONFIG_DIR / f"{self.role_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Agent config not found: {config_path}")
        with config_path.open() as f:
            data = yaml.safe_load(f)
        return AgentConfig(**data)

    def build(self) -> Agent:
        """Build and return a CrewAI Agent instance."""
        if self._agent is None:
            resolved_tools = [
                self.tool_registry.get(t) for t in self._config.tools if self.tool_registry.has(t)
            ]
            self._agent = Agent(
                role=self._config.role,
                goal=self._config.goal,
                backstory=self._config.backstory,
                tools=resolved_tools,
                llm=self.llm,
                verbose=self._config.verbose,
                allow_delegation=self._config.allow_delegation,
                max_iter=self._config.max_iter,
                memory=self._config.memory,
            )
        return self._agent

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} role={self._config.role!r}>"

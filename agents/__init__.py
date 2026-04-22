"""All 8 AI team agent class definitions."""

from agents.base_agent import BaseAgent


class ProductOwnerAgent(BaseAgent):
    """Supervisor agent — monitors progress, re-routes tasks, detects failures."""

    @property
    def role_name(self) -> str:
        return "product_owner"


class DataArchitectAgent(BaseAgent):
    """Planner agent — converts requirements into architectural blueprints."""

    @property
    def role_name(self) -> str:
        return "data_architect"


class DataEngineerAgent(BaseAgent):
    """Learner/tool operator — RAG flows, API ingestion, dataset management."""

    @property
    def role_name(self) -> str:
        return "data_engineer"


class DataScientistAgent(BaseAgent):
    """Thinker — algorithm selection, class imbalance strategy, model evaluation."""

    @property
    def role_name(self) -> str:
        return "data_scientist"


class MLEngineerAgent(BaseAgent):
    """Doer — training pipeline code generation, MLflow tracking, FastAPI deployment."""

    @property
    def role_name(self) -> str:
        return "ml_engineer"


class BackendEngineerAgent(BaseAgent):
    """Tool operator — external API integrations, Python script runner."""

    @property
    def role_name(self) -> str:
        return "backend_engineer"


class QATesterAgent(BaseAgent):
    """Critic — hallucination review, pytest generation and execution."""

    @property
    def role_name(self) -> str:
        return "qa_tester"


class CybersecurityAgent(BaseAgent):
    """Governance — policy compliance, toxicity checks, IBM Guardium hooks."""

    @property
    def role_name(self) -> str:
        return "cybersecurity"

"""CrewAI quick-start crew — wires all 8 agents for rapid prototyping.

Run:
    python orchestration/crewai_crew.py

Requires a .env file with WATSONX_API_KEY, WATSONX_PROJECT_ID.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from crewai import Crew, Process, Task
from dotenv import load_dotenv

from agents.registry import AgentRegistry
from config.quota_manager import QuotaManager
from tools.registry import ToolRegistry
from tools.watsonx_llm import build_watsonx_llm

load_dotenv()


def build_crew(project_description: str) -> Crew:
    """Build and return a CrewAI Crew for an ML project."""
    llm = build_watsonx_llm()
    tool_registry = ToolRegistry()
    agent_registry = AgentRegistry(llm=llm, tool_registry=tool_registry)

    # Build all agent instances
    supervisor = agent_registry.get("product_owner").build()
    architect = agent_registry.get("data_architect").build()
    engineer = agent_registry.get("data_engineer").build()
    scientist = agent_registry.get("data_scientist").build()
    ml_eng = agent_registry.get("ml_engineer").build()
    backend = agent_registry.get("backend_engineer").build()
    qa = agent_registry.get("qa_tester").build()
    security = agent_registry.get("cybersecurity").build()

    tasks = [
        Task(
            description=f"Review the project brief and define scope: {project_description}",
            expected_output="Scope document with deliverables and agent assignments.",
            agent=supervisor,
        ),
        Task(
            description="Produce an architectural plan: data flow, schemas, API contracts.",
            expected_output="Architecture document in Markdown.",
            agent=architect,
        ),
        Task(
            description="Ingest dataset from Kaggle. Clean and version the data.",
            expected_output="Path to cleaned dataset and ingestion report.",
            agent=engineer,
        ),
        Task(
            description="Select algorithm, handle class imbalance with SMOTE, define eval metrics.",
            expected_output="Algorithm selection report with justification and metric targets.",
            agent=scientist,
        ),
        Task(
            description="Generate training pipeline code, track experiment with MLflow.",
            expected_output="Training script and MLflow run ID.",
            agent=ml_eng,
        ),
        Task(
            description="Integrate FastAPI endpoint and wire external service dependencies.",
            expected_output="Working FastAPI serve module path and curl test examples.",
            agent=backend,
        ),
        Task(
            description="Review all outputs for hallucinations. Write and run pytest suite.",
            expected_output="QA report: issues found + pytest pass/fail summary.",
            agent=qa,
        ),
        Task(
            description="Scan generated code for OWASP vulnerabilities and policy violations.",
            expected_output="Security scan report with risk level and remediation steps.",
            agent=security,
        ),
    ]

    return Crew(
        agents=[supervisor, architect, engineer, scientist, ml_eng, backend, qa, security],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )


if __name__ == "__main__":
    quota = QuotaManager()
    quota.check_limits()

    description = (
        "Build an autonomous customer churn prediction system using a telecom dataset. "
        "Deploy as a FastAPI endpoint. Prioritise precision and recall over accuracy."
    )

    crew = build_crew(description)
    result = crew.kickoff()
    print("\n=== CREW RESULT ===\n", result)

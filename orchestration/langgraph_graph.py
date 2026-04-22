"""LangGraph production orchestration graph with stateful, cyclic execution.

The graph models the AI team as a directed graph where:
- Nodes = agent execution steps
- Edges = conditional transitions based on agent output state
- State = typed dict shared across all nodes (enables time-travel debugging)

Run:
    python orchestration/langgraph_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agents.registry import AgentRegistry
from tools.registry import ToolRegistry
from tools.watsonx_llm import build_watsonx_llm

load_dotenv()


# ---------------------------------------------------------------------------
# Shared state schema
# ---------------------------------------------------------------------------

class TeamState(TypedDict):
    project_description: str
    architecture_doc: str
    dataset_path: str
    algorithm_report: str
    training_run_id: str
    api_endpoint: str
    qa_report: str
    security_report: str
    messages: Annotated[list, add_messages]
    status: Literal["running", "needs_review", "complete", "failed"]


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------

def _make_node(agent_role: str, registry: AgentRegistry, output_key: str):
    """Create a LangGraph node function for a given agent role."""
    agent = registry.get(agent_role).build()

    def node_fn(state: TeamState) -> dict:
        task_input = state.get("project_description", "")
        result = agent.execute_task(task_input)  # type: ignore[attr-defined]
        return {output_key: result, "status": "running"}

    node_fn.__name__ = f"{agent_role}_node"
    return node_fn


def _supervisor_router(state: TeamState) -> Literal["data_architect", "qa_tester", END]:
    """Route based on current status — supervisor decides the next step."""
    if state.get("status") == "failed":
        return "qa_tester"  # Re-route to QA for remediation
    if state.get("qa_report") and state.get("security_report"):
        return END
    return "data_architect"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    llm = build_watsonx_llm()
    registry = AgentRegistry(llm=llm, tool_registry=ToolRegistry())

    graph = StateGraph(TeamState)

    # Register nodes
    graph.add_node("product_owner", _make_node("product_owner", registry, "messages"))
    graph.add_node("data_architect", _make_node("data_architect", registry, "architecture_doc"))
    graph.add_node("data_engineer", _make_node("data_engineer", registry, "dataset_path"))
    graph.add_node("data_scientist", _make_node("data_scientist", registry, "algorithm_report"))
    graph.add_node("ml_engineer", _make_node("ml_engineer", registry, "training_run_id"))
    graph.add_node("backend_engineer", _make_node("backend_engineer", registry, "api_endpoint"))
    graph.add_node("qa_tester", _make_node("qa_tester", registry, "qa_report"))
    graph.add_node("cybersecurity", _make_node("cybersecurity", registry, "security_report"))

    # Sequential pipeline edges
    graph.add_edge(START, "product_owner")
    graph.add_edge("data_architect", "data_engineer")
    graph.add_edge("data_engineer", "data_scientist")
    graph.add_edge("data_scientist", "ml_engineer")
    graph.add_edge("ml_engineer", "backend_engineer")
    graph.add_edge("backend_engineer", "qa_tester")
    graph.add_edge("qa_tester", "cybersecurity")
    graph.add_edge("cybersecurity", END)

    # Supervisor conditional routing (runs after product_owner)
    graph.add_conditional_edges(
        "product_owner",
        _supervisor_router,
        {
            "data_architect": "data_architect",
            "qa_tester": "qa_tester",
            END: END,
        },
    )

    return graph


def compile_graph(checkpointing: bool = True):
    """Compile the graph with optional in-memory checkpointing for time-travel debugging."""
    graph = build_graph()
    checkpointer = MemorySaver() if checkpointing else None
    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    app = compile_graph()
    config = {"configurable": {"thread_id": "churn-v1"}}
    initial_state: TeamState = {
        "project_description": (
            "Build an autonomous customer churn prediction system. "
            "Use telecom data. Prioritise precision and recall."
        ),
        "architecture_doc": "",
        "dataset_path": "",
        "algorithm_report": "",
        "training_run_id": "",
        "api_endpoint": "",
        "qa_report": "",
        "security_report": "",
        "messages": [],
        "status": "running",
    }
    final = app.invoke(initial_state, config=config)
    print("\n=== FINAL STATE ===")
    for k, v in final.items():
        if k != "messages":
            print(f"{k}: {v}")

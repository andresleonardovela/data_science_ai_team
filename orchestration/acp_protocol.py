"""Agent Communication Protocol (ACP) adapter — HTTP-based A2A communication.

Allows agents from different frameworks (LangChain, CrewAI, BeeAI) to
communicate via standard HTTP following the ACP specification.

Usage:
    server = ACPServer(host="0.0.0.0", port=9000)
    server.register_agent("data_scientist", my_agent_fn)
    server.run()  # Starts the FastAPI ACP endpoint

    # From another agent/service:
    client = ACPClient("http://localhost:9000")
    response = await client.call("data_scientist", {"task": "select algorithm"})
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# ACP message schema
# ---------------------------------------------------------------------------

class ACPMessage(BaseModel):
    agent_id: str
    task: str
    context: dict[str, Any] = {}
    sender: str = "orchestrator"


class ACPResponse(BaseModel):
    agent_id: str
    result: str
    status: str = "success"
    metadata: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# ACP Server
# ---------------------------------------------------------------------------

class ACPServer:
    """Exposes registered agent functions as HTTP endpoints."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9000) -> None:
        self.host = host
        self.port = port
        self._handlers: dict[str, Callable] = {}
        self.app = FastAPI(title="ACP Agent Server", version="1.0.0")
        self._register_routes()

    def register_agent(
        self, agent_id: str, handler: Callable[[ACPMessage], Coroutine | str]
    ) -> None:
        """Register an agent handler by ID."""
        self._handlers[agent_id] = handler

    def _register_routes(self) -> None:
        @self.app.post("/agents/{agent_id}/invoke", response_model=ACPResponse)
        async def invoke(agent_id: str, message: ACPMessage) -> ACPResponse:
            if agent_id not in self._handlers:
                raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not registered.")
            handler = self._handlers[agent_id]
            result = await asyncio.ensure_future(
                handler(message) if asyncio.iscoroutinefunction(handler)
                else asyncio.to_thread(handler, message)
            )
            return ACPResponse(agent_id=agent_id, result=str(result))

        @self.app.get("/agents")
        def list_agents() -> dict:
            return {"agents": list(self._handlers.keys())}

        @self.app.get("/health")
        def health() -> dict:
            return {"status": "ok"}

    def run(self) -> None:
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)


# ---------------------------------------------------------------------------
# ACP Client
# ---------------------------------------------------------------------------

class ACPClient:
    """Calls remote ACP agent endpoints over HTTP."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def call(
        self, agent_id: str, task: str, context: dict[str, Any] | None = None
    ) -> ACPResponse:
        message = ACPMessage(agent_id=agent_id, task=task, context=context or {})
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/agents/{agent_id}/invoke",
                json=message.model_dump(),
            )
            resp.raise_for_status()
            return ACPResponse(**resp.json())

    async def list_agents(self) -> list[str]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/agents")
            resp.raise_for_status()
            return resp.json().get("agents", [])

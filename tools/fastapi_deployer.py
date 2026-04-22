"""FastAPI model deployer tool — wraps a trained model as a REST endpoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class DeployerInput(BaseModel):
    serve_module: str = Field(
        description=(
            "Python module path to the FastAPI app, e.g. "
            "'projects.churn_prediction.src.serve:app'."
        )
    )
    host: str = Field(default="0.0.0.0", description="Host to bind the server to.")
    port: int = Field(default=8000, description="Port to bind the server to.")
    reload: bool = Field(default=False, description="Enable hot-reload (dev mode only).")


class FastAPIDeployerTool(BaseTool):
    """Starts a uvicorn server for a given FastAPI app module.

    Returns the server address. Note: this launches a subprocess and
    returns immediately — the caller is responsible for process management.
    """

    name: str = "fastapi_deployer"
    description: str = (
        "Deploy a trained ML model as a FastAPI REST endpoint by specifying its "
        "serve module path. Returns the server URL."
    )
    args_schema: type[BaseModel] = DeployerInput

    def _run(
        self,
        serve_module: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
    ) -> str:
        cmd = [
            sys.executable, "-m", "uvicorn",
            serve_module,
            "--host", host,
            "--port", str(port),
        ]
        if reload:
            cmd.append("--reload")

        try:
            # Launch as a detached background process
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return (
                f"Server started: http://{host}:{port}\n"
                f"Module: {serve_module}\n"
                f"Health check: http://{host}:{port}/health"
            )
        except Exception as exc:
            return f"Failed to start server: {exc}"


def build_deployer_tool() -> FastAPIDeployerTool:
    return FastAPIDeployerTool()

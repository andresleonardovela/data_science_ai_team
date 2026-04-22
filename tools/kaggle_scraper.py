"""Kaggle scraper tool — downloads datasets via the Kaggle API."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class _Settings(BaseSettings):
    kaggle_username: str
    kaggle_key: str

    model_config = {"env_file": ".env", "extra": "ignore"}


class KaggleInput(BaseModel):
    dataset: str = Field(
        description=(
            "Kaggle dataset identifier in 'owner/dataset-name' format, "
            "e.g. 'blastchar/telco-customer-churn'."
        )
    )
    output_dir: str = Field(
        default="projects/churn_prediction/data/raw",
        description="Local directory to download the dataset into.",
    )


class KaggleScraperTool(BaseTool):
    """Downloads a Kaggle dataset to a local directory."""

    name: str = "kaggle_scraper"
    description: str = (
        "Download a dataset from Kaggle by its identifier (owner/dataset-name). "
        "Returns the path to the downloaded files."
    )
    args_schema: type[BaseModel] = KaggleInput

    def _run(self, dataset: str, output_dir: str = "projects/churn_prediction/data/raw") -> str:
        settings = _Settings()
        env = os.environ.copy()
        env["KAGGLE_USERNAME"] = settings.kaggle_username
        env["KAGGLE_KEY"] = settings.kaggle_key

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_path), "--unzip"],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )

        if result.returncode != 0:
            return f"Kaggle download failed:\n{result.stderr}"

        files = list(out_path.glob("**/*"))
        file_list = "\n".join(str(f) for f in files if f.is_file())
        return f"Download complete. Files in {out_path}:\n{file_list}"


def build_kaggle_tool() -> KaggleScraperTool:
    return KaggleScraperTool()

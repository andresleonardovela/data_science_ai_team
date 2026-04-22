"""Unified IBM Granite LLM wrapper (LangChain-compatible)."""

from __future__ import annotations

import os

from langchain_ibm import WatsonxLLM
from pydantic_settings import BaseSettings


class WatsonxSettings(BaseSettings):
    watsonx_api_key: str
    watsonx_project_id: str
    watsonx_url: str = "https://us-south.ml.cloud.ibm.com"
    watsonx_model_id: str = "ibm/granite-13b-instruct-v2"

    model_config = {"env_file": ".env", "extra": "ignore"}


def build_watsonx_llm(
    model_id: str | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
) -> WatsonxLLM:
    """Build and return a configured WatsonxLLM instance.

    Reads credentials from environment variables / .env file.
    Temperature is kept low (0.1) for deterministic agent reasoning.
    """
    settings = WatsonxSettings()
    return WatsonxLLM(
        model_id=model_id or settings.watsonx_model_id,
        url=settings.watsonx_url,
        project_id=settings.watsonx_project_id,
        apikey=settings.watsonx_api_key,
        params={
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "decoding_method": "greedy",
            "repetition_penalty": 1.1,
        },
    )

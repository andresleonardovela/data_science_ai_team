"""RAG tool — LangChain retriever backed by watsonx.ai embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crewai.tools import BaseTool
from langchain_community.vectorstores import FAISS
from langchain_ibm import WatsonxEmbeddings
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class _Settings(BaseSettings):
    watsonx_api_key: str
    watsonx_project_id: str
    watsonx_url: str = "https://us-south.ml.cloud.ibm.com"

    model_config = {"env_file": ".env", "extra": "ignore"}


class RAGInput(BaseModel):
    query: str = Field(description="Natural language query to retrieve relevant documents for.")


class RAGTool(BaseTool):
    """Retrieves relevant document chunks using watsonx.ai embeddings + FAISS."""

    name: str = "rag_tool"
    description: str = (
        "Search an indexed knowledge base for information relevant to a query. "
        "Use this before generating an answer that requires external knowledge."
    )
    args_schema: type[BaseModel] = RAGInput
    vectorstore: Any = None

    def _run(self, query: str) -> str:
        if self.vectorstore is None:
            return "RAG index not initialised. Call load_index() before using this tool."
        docs = self.vectorstore.similarity_search(query, k=4)
        return "\n\n---\n\n".join(d.page_content for d in docs)

    def load_index(self, index_path: str | Path) -> None:
        """Load a pre-built FAISS index from disk."""
        settings = _Settings()
        embeddings = WatsonxEmbeddings(
            model_id="ibm/slate-125m-english-rtrvr",
            url=settings.watsonx_url,
            project_id=settings.watsonx_project_id,
            apikey=settings.watsonx_api_key,
        )
        self.vectorstore = FAISS.load_local(
            str(index_path), embeddings, allow_dangerous_deserialization=True
        )


def build_rag_tool() -> RAGTool:
    return RAGTool()

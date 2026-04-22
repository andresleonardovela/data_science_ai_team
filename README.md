# Data Science AI Team

An **AI-agent-based machine learning ecosystem** where a team of 8 specialized autonomous agents collaborates end-to-end — from dataset ingestion and feature engineering to model training, deployment, QA testing, and security auditing — powered by **IBM watsonx.ai (Granite models)**, **LangGraph**, and **CrewAI**.

[![CI](https://github.com/andresleonardovela/data_science_ai_team/actions/workflows/ci.yml/badge.svg)](https://github.com/andresleonardovela/data_science_ai_team/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

This project implements a **multi-agent AI system** that mimics a real data science delivery team. Each agent has a well-defined role, communicates via structured protocols, and collectively executes ML pipelines autonomously.

```
User Request
     │
     ▼
Product Owner (supervisor)
     ├── Data Architect      → architecture & schema design
     ├── Data Engineer       → ingestion, cleaning, feature engineering
     ├── Data Scientist      → EDA, algorithm selection, experimentation
     ├── ML Engineer         → model training, MLflow tracking, deployment
     ├── Backend Engineer    → FastAPI serving layer
     ├── QA Tester           → evaluation gates & regression checks
     └── Cybersecurity       → OWASP scanning & governance audit
```

---

## Key Features

| Feature | Details |
|---|---|
| **8 Autonomous Agents** | YAML-driven configs, shared BaseAgent ABC, AgentRegistry |
| **Dual Orchestration** | CrewAI (prototyping) + LangGraph stateful graph (production) |
| **IBM watsonx.ai** | Granite 13B Instruct via LangChain-IBM; free-tier quota tracking |
| **ACP Protocol** | HTTP-based Agent Communication Protocol via FastAPI + httpx |
| **MCP Config** | Model Context Protocol connecting agents to tools & resources |
| **RAG Tool** | FAISS vector store with IBM Slate 125M embeddings |
| **Governance** | Drift detection, bias monitoring, toxicity/PII filtering |
| **CI/CD** | GitHub Actions matrix (Python 3.11 + 3.12), ruff, mypy, pytest-cov |

---

## Project Structure

```
data_science_ai_team/
├── agents/                    # 8 agent classes + YAML configs
│   ├── config/                # Per-agent role/goal/backstory YAML
│   ├── base_agent.py          # Abstract base class (Pydantic config + CrewAI wiring)
│   └── registry.py            # Lazy-loading AgentRegistry
├── config/
│   └── quota_manager.py       # IBM Lite free-tier quota tracker (300K tokens/month)
├── governance/
│   ├── guardrails.py          # Drift + bias monitoring (watsonx.governance hooks)
│   └── toxicity_filter.py     # Pre/post LLM call toxicity & PII filter
├── orchestration/
│   ├── crewai_crew.py         # Quick-start sequential CrewAI crew
│   ├── langgraph_graph.py     # Production stateful graph with MemorySaver
│   ├── acp_protocol.py        # HTTP Agent Communication Protocol (ACP)
│   └── mcp_config.yaml        # Model Context Protocol configuration
├── projects/
│   └── churn_prediction/      # End-to-end reference project (Telco churn)
│       └── src/               # ingest, preprocess, train, evaluate, serve
├── tools/
│   ├── watsonx_llm.py         # IBM Granite LLM wrapper
│   ├── rag_tool.py            # FAISS-based RAG tool
│   ├── kaggle_scraper.py      # Kaggle dataset downloader
│   ├── fastapi_deployer.py    # Uvicorn-based model server launcher
│   ├── guardium_scanner.py    # OWASP static scan + IBM Guardium hook
│   └── registry.py            # ToolRegistry (lazy discovery)
├── tests/                     # Unit tests (governance, tools, agents)
├── .github/workflows/ci.yml   # GitHub Actions CI pipeline
├── .env.example               # Credential template (never commit .env)
└── pyproject.toml             # Hatchling package + all dependencies
```

---

## Quickstart

### 1. Clone & set up environment

```bash
git clone https://github.com/andresleonardovela/data_science_ai_team.git
cd data_science_ai_team

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in your IBM Cloud & Kaggle credentials
```

Required variables:

| Variable | Description |
|---|---|
| `WATSONX_API_KEY` | IBM Cloud API key |
| `WATSONX_PROJECT_ID` | watsonx.ai project ID |
| `WATSONX_URL` | Regional endpoint (default: `https://us-south.ml.cloud.ibm.com`) |
| `WATSONX_MODEL_ID` | Model to use (default: `ibm/granite-13b-instruct-v2`) |
| `KAGGLE_USERNAME` | Kaggle username for dataset downloads |
| `KAGGLE_KEY` | Kaggle API key |

### 3. Run the reference project (churn prediction)

```bash
# Orchestrate via LangGraph (production)
python orchestration/langgraph_graph.py

# Or via CrewAI (quick prototype)
python orchestration/crewai_crew.py
```

### 4. Serve the trained model

```bash
uvicorn projects.churn_prediction.src.serve:app --reload
# POST http://localhost:8000/predict
```

---

## Reference Project — Telco Churn Prediction

The `projects/churn_prediction/` directory is a complete end-to-end example:

1. **Ingest** — Downloads `blastchar/telco-customer-churn` from Kaggle
2. **Preprocess** — Cleans data, label-encodes, one-hot encodes, scales, applies SMOTE
3. **Train** — Random Forest (300 trees, balanced class weight) with 5-fold CV + MLflow tracking
4. **Evaluate** — QA gate: recall ≥ 0.70, precision ≥ 0.55, F1 ≥ 0.60, ROC-AUC ≥ 0.75
5. **Serve** — FastAPI endpoint returning `churn`, `churn_probability`, and `risk_level` (HIGH/MEDIUM/LOW)

---

## IBM watsonx.ai Free Tier

This project is optimized to run within **IBM Cloud Lite limits**:

| Resource | Monthly Limit |
|---|---|
| Tokens | 300,000 |
| Watson Orchestrate Actions | 100 |
| Compute Unit Hours (CUH) | 20 |

Quota is tracked automatically in `config/quota_manager.py` and persisted to `.quota_usage.json` (gitignored). A Rich summary table is printed before each crew run.

---

## Running Tests

```bash
pytest tests/ projects/ --cov=. -v
```

The test suite covers:
- Governance: toxicity filter, PII redaction, drift/bias monitors
- Tools: Guardium scanner patterns, ToolRegistry discovery
- Churn pipeline: preprocessing shapes, SMOTE balance, FastAPI endpoints, QA thresholds

---

## Architecture Decisions

- **LangGraph** is used for production orchestration — its stateful `StateGraph` with `MemorySaver` enables time-travel debugging and fault-tolerant resumption.
- **CrewAI** provides a rapid prototyping path with minimal boilerplate.
- **ACP (Agent Communication Protocol)** over HTTP allows agents to be deployed as independent microservices.
- **YAML-driven agent configs** decouple role definitions from code, enabling non-engineers to tune agent behavior.
- **Pydantic-settings** enforces strict environment variable validation at startup, preventing silent misconfigurations.

---

## Contributing

1. Fork the repo and create a feature branch from `main`
2. Run `ruff check .` and `pytest` before opening a PR
3. PRs to `main` require CI to pass on both Python 3.11 and 3.12

---

## License

MIT © Andres Leonardo Vela

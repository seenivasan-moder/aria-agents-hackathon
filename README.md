<div align="center">

# AIRIA SENTINEL

### Agentic OS — Multi-Agent Intelligence Platform

**Airia AI Agents Challenge — Track 2: Active Agents**

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)](https://python.org)
[![Airia SDK](https://img.shields.io/badge/Airia_SDK-0.1.41-purple?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCI+PHBhdGggZD0iTTEyIDJMMiAyMmgyMEwxMiAyeiIgZmlsbD0id2hpdGUiLz48L3N2Zz4=)](https://platform.airia.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Evaluations](https://img.shields.io/badge/Airia_Eval-30%2F30_passed-brightgreen)](https://airia.ai/evaluations)

<br/>

An **Operating System for AI Agents** — orchestrating the collaboration between humans, local LLMs, and cloud AI to automate critical enterprise processes with **full traceability** and **human-in-the-loop control**.

**4 agent groups** | **12+ specialized agents** | **5-GPU local cluster (43GB VRAM)** | **Airia pipelines** | **3-way multi-model consensus** | **HITL approval** | **SQLite audit trail**

<br/>

[Quick Start](#-quick-start) &bull; [Architecture](#-architecture) &bull; [Agent Groups](#-agent-groups) &bull; [Demo](#-demo-mode) &bull; [Evaluation](#-airia-evaluation-results) &bull; [For Judges](#-for-judges--compliance)

</div>

---

## Vision

Airia Sentinel is not just a treasury tool — it's an **Agentic OS**: a domain-agnostic platform where **groups of specialized AI agents** collaborate to solve complex problems. Each agent group follows the same proven pattern:

```
Parallel Data Collection  →  Multi-Model Consensus  →  Structured Output  →  Human Approval
```

The platform currently ships with **4 agent groups**, each solving a different real-world problem:

| Group | Domain | Agents | Status |
|-------|--------|--------|--------|
| **Sentinel** | Treasury Risk Management | 4 agents | Production |
| **Meta-Exchange** | Multi-AI Provider Orchestration | 3 agents | Production |
| **Organizer** | Intelligent File Management | 2 agents | Production |
| **JARVIS** | Personal AI Assistant | 2 agents | Production |

---

## For Judges & Compliance

### For Technical Reviewers

- **Airia SDK v0.1.41**: 4 pipelines, 4 deployments, 3 custom tools, dual execution mode (pipeline + temporary assistant)
- **Local AI Cluster**: 5 GPUs, 43GB VRAM, qwen3-30b permanent, qwen3:1.7b for fast inference
- **3-Way Consensus**: LM Studio + Ollama + Airia pipeline — weighted confidence averaging with dissent detection
- **Evaluation**: 30/30 test cases passed on Airia platform, Sentinel-3 at 71.66% precision
- **Modern Stack**: Python 3.13, async/await, Pydantic v2, httpx connection pooling, FastAPI

### For Business & Legal Reviewers

- **Immutable Audit Trail**: Every AI decision is logged in SQLite with timestamp, model used, input/output summary, and latency. No decision can be taken without a trace.
- **Human-in-the-Loop (HITL)**: No strategy is executed without explicit human approval via FastAPI webhook. Supports approve, reject, and escalate workflows.
- **Accountability**: Each agent logs which AI model produced which output. Cross-model consensus ensures no single model can introduce bias unchecked.
- **Regulatory Readiness**: PDF reports are generated in Big Four style with executive summary, market analysis, exposure breakdown, and audit certification — ready for board review.
- **Data Privacy**: Hybrid architecture processes sensitive data locally first (5-GPU cluster), only sending anonymized summaries to cloud AI for enrichment.

### For End Users

- **One Command**: `uv run python main.py` runs the entire pipeline
- **Professional PDF Reports**: A4 reports with tables, charts, and Airia-generated executive summaries
- **Natural Language**: All agent prompts are in French, outputs are structured and readable
- **Real-Time Data**: Live crypto prices from MEXC, simulated forex/commodities for demo
- **Cinematic Demo**: Built-in 7-step demo mode for presentations

---

## The Problem

Enterprise teams face fragmented workflows: treasury managers juggle multiple tools for market monitoring, developers switch between AI providers without cost control, IT teams manage terabytes of unclassified files. **Every disconnected process is a risk**.

> *"In Q1 2026, ACME Corp's JPY exposure of 350M went unhedged for 72 hours during a BOJ policy shift, resulting in a $2.3M unrealized loss."*

## The Solution

Airia Sentinel provides **specialized agent groups** for each domain, all orchestrated through Airia pipelines with the same proven architecture:

### Sentinel Group — Treasury Risk



| Phase | Agent | Role | Intelligence Source |
|:-----:|-------|------|-------------------|
| 1 | **Market Intelligence** | Scans crypto, forex, and commodity markets for risk signals | CCXT (MEXC) + Airia Pipeline |
| 1 | **Corporate Context** | Analyzes internal financial exposure and concentration risks | Local calc + Airia Pipeline |
| 2 | **Consensus & Strategy** | Generates hedging strategies via 3-way multi-IA consensus | LM Studio + Ollama + Airia |
| 3 | **Compliance & Docs** | Produces professional PDF report + audit trail + HITL approval | ReportLab + Airia Pipeline |

> Phase 1 runs in **parallel**. Phases 2 and 3 run **sequentially**. All steps produce a full **audit trail** in SQLite.

---

## Architecture

```
                         ┌─────────────────────────┐
                         │     AIRIA PLATFORM       │
                         │  4 Pipelines + 3 Tools   │
                         │  + Evaluation Suite       │
                         └────────────┬──────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
   ┌──────────▼──────────┐  ┌────────▼────────┐  ┌──────────▼──────────┐
   │   Agent 1            │  │   Agent 2        │  │   Agent 3            │
   │   Market             │  │   Corporate      │  │   Consensus          │
   │   Intelligence       │  │   Context        │  │   & Strategy         │
   │                      │  │                  │  │                      │
   │   CCXT + Airia       │  │   Local + Airia  │  │   M1 + OL1 + Airia   │
   │   10 pairs live      │  │   5 currencies   │  │   3-way voting        │
   └──────────┬───────────┘  └────────┬─────────┘  └──────────┬───────────┘
              │ Phase 1 (parallel)    │                       │ Phase 2
              └───────────────────────┼───────────────────────┘
                                      │
                           ┌──────────▼──────────┐
                           │   Agent 4            │
                           │   Compliance & Docs  │    Phase 3
                           │                      │
                           │   PDF + Audit Trail  │
                           │   + Airia Summary     │
                           └──────────┬───────────┘
                                      │
                           ┌──────────▼──────────┐
                           │   HITL Gateway       │
                           │   FastAPI Webhook    │
                           │                      │
                           │   Approve / Reject   │
                           │   / Escalate         │
                           └──────────────────────┘
```

### Hybrid Execution Model

Each agent operates in **hybrid mode** — local computation first, then Airia pipeline enrichment:

| Layer | Purpose | Benefit |
|-------|---------|---------|
| **Local First** | LM Studio cluster (5 GPU, 43GB VRAM) + Ollama | Speed + offline resilience |
| **Cloud Enrichment** | Airia pipelines with GPT-5.1 | Quality + formal analysis |
| **Consensus Merge** | Weighted averaging of confidence scores | Reliability + dissent detection |

This ensures the system works **fully offline** with the local AI cluster, while Airia adds **cloud-powered enrichment** when available.

---

## Tech Stack

| Component | Technology | Details |
|-----------|------------|---------|
| AI Orchestration | **Airia SDK v0.1.41** | 4 pipelines + 4 deployments + 3 tools |
| Local AI (Deep) | **LM Studio** | qwen3-30b, 5 GPUs, 43GB VRAM |
| Local AI (Light) | **Ollama** | qwen3:1.7b, fast inference |
| Market Data | **CCXT v4+** | MEXC exchange, multi-pair real-time |
| Data Models | **Pydantic v2** | Strict validation, JSON serialization |
| PDF Reports | **ReportLab v4** | Professional A4 reports with tables |
| HITL Webhook | **FastAPI + Uvicorn** | Approve / reject / escalate strategies |
| Database | **SQLite (WAL mode)** | 5 tables, full audit trail |
| CLI UI | **Rich v14** | Tables, panels, progress bars |
| Package Manager | **uv v0.10** | Fast Python dependency management |
| Language | **Python 3.13** | Modern async/await patterns |

---

## Agent Groups

### Group 1: Sentinel — Treasury Risk (4 agents)

The flagship use case. Scans markets, analyzes corporate exposure, generates hedging strategies via 3-way consensus, and produces audit-ready PDF reports.

> See pipeline details in the [Architecture](#architecture) section above.

### Group 2: Meta-Exchange — Multi-AI Orchestration (3 agents)

When you use **multiple AI providers** simultaneously (LM Studio, Ollama, Claude, GPT), who manages the costs, quality, and routing?

| Agent | File | Role |
|-------|------|------|
| **Meta-Router** | `src/agents/meta/meta_router.py` | Analyzes query complexity, estimates costs, routes to optimal provider (M1/M2/OL1/Cloud/Airia) |
| **Context-Manager** | `src/agents/meta/context_manager.py` | Manages cross-provider memory, deduplicates queries, tracks token budgets per session |
| **Quality-Auditor** | `src/agents/meta/quality_auditor.py` | Compares responses across providers, detects hallucinations, scores coherence/completeness |

```
User Query → Meta-Router (classify + route) → Provider (M1/OL1/Cloud)
                                                      ↓
                              Quality-Auditor (validate) ← Context-Manager (dedup + memory)
```

### Group 3: Organizer — Intelligent File Management (2 agents)

AI-powered disk intelligence: scan, classify, detect sensitive files, and identify duplicates.

| Agent | File | Role |
|-------|------|------|
| **Librarian** | `src/agents/organizer/librarian.py` | Scans directories, classifies files (code/doc/image/data/config), detects API keys and credentials |
| **Deduplicator** | `src/agents/organizer/dedup_agent.py` | Finds duplicate files by content hash, recommends keep/delete, calculates recoverable space |

```
Target Dir → Librarian (scan + classify + sensitivity) → Deduplicator (hash + recover space)
```

### Group 4: JARVIS — Personal AI Assistant (2 agents)

Multi-agent personal assistant with intent classification and intelligent sub-agent routing.

| Agent | File | Role |
|-------|------|------|
| **Intent-Classifier** | `src/agents/jarvis/intent_classifier.py` | Parses voice/text input, detects domain (system/web/trading/analysis), extracts entities |
| **Execution-Engine** | `src/agents/jarvis/execution_engine.py` | Routes intents to sub-agents (ia-deep/ia-fast/ia-system/ia-trading), handles timeouts and fallback chains |

```
"Ouvre Chrome" → Intent-Classifier (system/open, app=chrome) → Execution-Engine → ia-system
"Analyse ce code" → Intent-Classifier (analysis/analyze) → Execution-Engine → ia-deep (qwen3-30b)
```

> See [docs/USE_CASES.md](docs/USE_CASES.md) for detailed architecture and agent descriptions for each group.

---

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Airia API key ([platform.airia.com](https://platform.airia.com))
- *(Optional)* LM Studio with a loaded model
- *(Optional)* Ollama with qwen3:1.7b
- *(Optional)* MEXC API credentials for live market data

### Installation

```bash
# Clone the repository
git clone https://github.com/Turbo31150/aria-agents-hackathon-private.git
cd aria-agents-hackathon-private

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Configuration (.env)

```env
# Required
AIRIA_API_KEY=your_airia_api_key_here

# Optional: Pre-configured pipeline IDs (auto-created by setup script)
AIRIA_MARKET_PIPELINE_ID=
AIRIA_CORPORATE_PIPELINE_ID=
AIRIA_CONSENSUS_PIPELINE_ID=
AIRIA_COMPLIANCE_PIPELINE_ID=

# Optional: LM Studio cluster
LM_STUDIO_URL=http://127.0.0.1:1234

# Optional: Ollama
OLLAMA_URL=http://127.0.0.1:11434

# Optional: MEXC for live crypto data
MEXC_API_KEY=
MEXC_SECRET_KEY=

# HITL webhook port
HITL_PORT=8900
```

### Setup Airia Pipelines (One-Time)

```bash
# Create the 4 agent pipelines on the Airia platform
uv run python setup_airia_pipelines.py

# This outputs pipeline IDs — add them to your .env
```

> **Note**: If no pipeline IDs are configured, the system uses Airia's `execute_temporary_assistant()` with embedded prompts. No setup required.

---

## Usage

### Full Pipeline (Default)

Run the complete 4-agent pipeline: scan markets, analyze exposure, generate consensus strategies, produce PDF report.

```bash
uv run python main.py
# or
uv run python main.py pipeline
```

**Output:**
- Rich terminal dashboard with tables for each agent
- PDF report in `data/reports/sentinel_report_<run_id>.pdf`
- SQLite audit trail in `data/sentinel.db`
- HITL approval request (pending)

### Market Scan Only

```bash
uv run python main.py scan
```

### System Status

```bash
uv run python main.py status
```

### Demo Mode

Cinematic walkthrough with narration panels, step-by-step agent execution, and HITL simulation. Designed for a 3-4 minute screen recording.

```bash
uv run python main.py demo
```

### HITL Webhook Server

```bash
uv run python main.py hitl
```

---

## Demo Mode

The built-in demo mode provides a **cinematic 7-step walkthrough** designed for hackathon video recording:

| Step | Scene | Duration |
|------|-------|----------|
| 1 | Problem / Solution introduction | ~5s |
| 2 | Scenario: JPY volatility spike at ACME Corp | ~3s |
| 3 | Architecture overview (pipeline flow diagram) | ~3s |
| 4 | Infrastructure health check (cluster + Airia status) | ~2s |
| 5 | Full pipeline execution with narration (Phase 1-2-3) | ~60s |
| 6 | HITL approval simulation (CFO approves strategy) | ~5s |
| 7 | Summary with key metrics + tech stack | ~4s |

```bash
uv run python main.py demo
```

---

## Airia Evaluation Results

The project has been evaluated on the Airia platform with **30/30 test cases passing**:

| Agent | Score | Latency | Tokens | Cost | Precision |
|-------|-------|---------|--------|------|-----------|
| **Sentinel-3: Consensus Strategy** | 1.00 | 9.00s | 978.8 | $0.007739 | **71.66%** |
| **Sentinel-1: Market Intelligence** | 1.00 | 1.13s | 249.5 | $0.000612 | 2.00% |
| **Sentinel-2: Corporate Context** | 1.00 | 1.12s | 217.8 | $0.000601 | 0.00% |

- **Evaluation model**: mistral-small-latest
- **Total cost**: $0.089520
- **Execution**: 30/30 completed

> See [EVALUATION.md](EVALUATION.md) for detailed analysis and interpretation of results.

---

## HITL API Reference

The FastAPI webhook server enables **human-in-the-loop approval** for generated strategies:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/pending` | List pending approvals |
| `POST` | `/approve` | Approve a strategy |
| `POST` | `/reject` | Reject and escalate |
| `GET` | `/report/{run_id}` | Get full run report |

**Approve a strategy:**
```bash
curl -X POST http://localhost:8900/approve \
  -H "Content-Type: application/json" \
  -d '{"run_id": "abc123", "strategy_name": "Conservative", "approved_by": "CFO"}'
```

**Reject a strategy:**
```bash
curl -X POST http://localhost:8900/reject \
  -H "Content-Type: application/json" \
  -d '{"run_id": "abc123", "strategy_name": "Aggressive", "rejected_by": "CFO", "reason": "Too risky for Q2"}'
```

---

## Database Schema

SQLite with WAL mode, 5 tables — every pipeline step is fully traceable:

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `signals` | Market scan results | symbol, price, risk_score, direction, regime |
| `exposures` | Corporate exposure | company_id, total_assets, risk_score, positions |
| `strategies` | Hedging strategies | name, confidence, cost, risk_reduction |
| `approvals` | HITL approvals | strategy_name, status, approved_by |
| `audit_log` | Full audit trail | step, agent, model_used, latency_ms |

All tables indexed by `run_id` for fast pipeline queries.

---

## Project Structure

```
aria-agents-hackathon-private/
├── main.py                              # CLI entry point (5 modes)
├── pyproject.toml                       # Dependencies
├── setup_airia_pipelines.py             # One-time Airia pipeline provisioning
├── .env.example                         # Template
│
├── src/
│   ├── config.py                        # Centralized config (Airia, cluster, markets)
│   ├── models.py                        # Pydantic models (9 models, strict validation)
│   ├── database.py                      # SQLite audit trail (5 tables, WAL mode)
│   ├── orchestrator.py                  # Main pipeline orchestration (3 phases)
│   ├── airia_bridge.py                  # Airia SDK wrapper (dual mode + 4 FR prompts)
│   │
│   ├── agents/
│   │   │── # Sentinel Group (Treasury Risk)
│   │   ├── market_intelligence.py       # Agent: CCXT scan + Airia enrichment
│   │   ├── corporate_context.py         # Agent: Treasury analysis + Airia risks
│   │   ├── consensus_strategy.py        # Agent: 3-way multi-IA consensus
│   │   ├── compliance_docs.py           # Agent: PDF report + Airia summary
│   │   │
│   │   │── # Meta-Exchange Group (AI Orchestration)
│   │   ├── meta/
│   │   │   ├── meta_router.py           # Agent: Intelligent AI provider routing
│   │   │   ├── context_manager.py       # Agent: Cross-provider memory + dedup
│   │   │   └── quality_auditor.py       # Agent: Response quality validation
│   │   │
│   │   │── # Organizer Group (File Management)
│   │   ├── organizer/
│   │   │   ├── librarian.py             # Agent: File scan + classification
│   │   │   └── dedup_agent.py           # Agent: Deduplication + space recovery
│   │   │
│   │   │── # JARVIS Group (Personal Assistant)
│   │   └── jarvis/
│   │       ├── intent_classifier.py     # Agent: Intent detection + entity extraction
│   │       └── execution_engine.py      # Agent: Sub-agent routing + fallback chains
│   │
│   ├── services/
│   │   ├── market_data.py               # CCXT + simulated Forex/Commodities
│   │   ├── lm_cluster.py               # LM Studio + Ollama interface
│   │   ├── pdf_generator.py            # ReportLab PDF generation
│   │   └── hitl_webhook.py             # FastAPI HITL server
│   │
│   └── utils/
│       ├── http_pool.py                 # Async HTTP connection pool (httpx)
│       └── retry.py                     # Retry with exponential backoff
│
├── demo/
│   └── demo_scenario.py                 # Cinematic demo script (7 steps)
│
├── data/
│   ├── mock_corporate.json              # Simulated ACME Corp treasury data
│   ├── sentinel.db                      # SQLite database (auto-created)
│   └── reports/                         # Generated PDF reports
│
├── docs/
│   ├── USE_CASES.md                     # 4 agent groups with architectures
│   ├── ARCHITECTURE.md                  # Technical deep dive (Mermaid diagrams)
│   └── EVALUATION.md                    # Airia evaluation results & analysis
│
└── launchers/
    └── SENTINEL.bat                     # Windows launcher
```

---

## Hackathon Criteria Mapping

| Criteria | Our Response |
|----------|-------------|
| **Technical Implementation** | Airia SDK (4 pipelines + deployments) + Local AI Cluster (5 GPU, 43GB VRAM) + 12+ agents across 4 groups + CCXT live data + 3-way Multi-IA Consensus + FastAPI HITL + SQLite audit trail |
| **UX/UI Design** | Rich CLI dashboard + Professional PDF Reports (Big Four style) + HITL REST API + Cinematic demo mode + French prompts |
| **Potential Impact** | Domain-agnostic Agentic OS: Treasury ($1T+ market), AI governance (cost control), file management, personal AI. Reduces decision latency from days to minutes. |
| **Creativity/Uniqueness** | Only solution combining local LLM cluster (5 GPU) + cloud Airia platform + multi-model consensus + 4 distinct agent groups showing platform versatility. Not just one use case — a complete agent operating system. |

---

## Performance

| Agent | Latency | Sources |
|-------|---------|---------|
| Market Intelligence | ~4s | CCXT + Airia pipeline |
| Corporate Context | ~3s | Local calc + Airia pipeline |
| Consensus & Strategy | ~35s | M1 (qwen3-30b) + OL1 (qwen3:1.7b) + Airia |
| Compliance & Docs | ~21s | ReportLab + Airia executive summary |
| **Total Pipeline** | **~65s** | Full end-to-end |

---

## License

MIT

---

<div align="center">

**Built with [Airia](https://airia.ai)** &bull; Agentic OS — Multi-Agent Intelligence Platform

*4 Agent Groups &bull; 12+ Specialized Agents &bull; 5-GPU Cluster &bull; 3-Way Consensus &bull; Full Audit Trail*

*Hackathon submission by [Franck Delmas](https://github.com/Turbo31150)*

</div>

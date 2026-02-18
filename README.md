# Airia Sentinel

**Multi-Agent Treasury Orchestration & Risk Management**

> Airia AI Agents Challenge - Track 2: Active Agents

---

## Problem

Enterprise treasury teams face a constant challenge: monitoring global markets (Forex, Crypto, Commodities), assessing internal financial exposure, and making timely hedging decisions. Today, this process is fragmented across multiple tools, slow, and prone to human error. A single missed currency spike can cost millions.

## Solution

Airia Sentinel is a **multi-agent AI system** that orchestrates the entire treasury risk management workflow in real-time:

1. **Market Intelligence Agent** — scans crypto, forex, and commodity markets for risk signals
2. **Corporate Context Agent** — analyzes internal financial exposure and concentration risks
3. **Consensus & Strategy Agent** — generates hedging strategies using multi-IA consensus (3 AI models in parallel)
4. **Compliance & Docs Agent** — produces a professional PDF report with full audit trail + HITL approval

All orchestrated through **Airia pipelines** with **Human-in-the-Loop (HITL)** approval via webhook.

---

## Architecture

```
                    +---------------------+
                    |   AIRIA PLATFORM    |
                    |   (4 Pipelines)     |
                    +---------+-----------+
                              |
            +-----------------+------------------+
            |                 |                  |
    +-------v-------+ +------v------+ +---------v-------+
    |  Agent 1      | |  Agent 2    | |  Agent 3        |
    |  Market       | |  Corporate  | |  Consensus      |
    |  Intelligence | |  Context    | |  & Strategy     |
    |  [CCXT+Airia] | |  [Local+AI] | |  [M1+OL1+Airia] |
    +-------+-------+ +------+------+ +---------+-------+
            |                |                   |
            | Phase 1        | Phase 1           | Phase 2
            | (Parallel)     | (Parallel)        | (Sequential)
            +----------------+-------------------+
                             |
                    +--------v--------+
                    |   Agent 4       |
                    |   Compliance    | Phase 3
                    |   & Docs        |
                    |   [PDF+Audit]   |
                    +--------+--------+
                             |
                    +--------v--------+
                    |   HITL          |
                    |   Webhook       |
                    |   (FastAPI)     |
                    +-----------------+
```

### Pipeline Flow

| Phase | Agents | Mode | Description |
|-------|--------|------|-------------|
| Phase 1 | Agent 1 + Agent 2 | **Parallel** | Market scan + Corporate analysis run simultaneously |
| Phase 2 | Agent 3 | **Sequential** | 3-way consensus: LM Studio (qwen3-30b) + Ollama (qwen3:1.7b) + Airia pipeline |
| Phase 3 | Agent 4 | **Sequential** | PDF report generation + Airia executive summary + HITL approval request |

### Hybrid Execution Model

Each agent runs in **hybrid mode**: local computation first, then Airia pipeline enrichment. This ensures:
- **Resilience**: works fully offline with local AI cluster
- **Quality**: Airia pipeline adds cloud-powered analysis on top
- **Speed**: local results appear fast, Airia enriches asynchronously

---

## Tech Stack

| Component | Technology | Details |
|-----------|------------|---------|
| AI Orchestration | Airia SDK v0.1.41 | 4 pipelines + 4 deployments + 3 tools |
| Local AI (Deep) | LM Studio | qwen3-30b, 5 GPUs, 43GB VRAM |
| Local AI (Light) | Ollama | qwen3:1.7b, fast inference |
| Market Data | CCXT v4+ | MEXC exchange, multi-pair |
| Data Models | Pydantic v2 | Strict validation, JSON serialization |
| PDF Reports | ReportLab v4 | Professional A4 reports with tables |
| HITL Webhook | FastAPI + Uvicorn | Approve/reject/escalate strategies |
| Database | SQLite (WAL mode) | 5 tables, full audit trail |
| CLI UI | Rich v14 | Tables, panels, progress bars |
| Package Manager | uv v0.10 | Fast Python dependency management |
| Language | Python 3.13 | Modern async/await patterns |

---

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Airia API key ([platform.airia.com](https://platform.airia.com))
- (Optional) LM Studio with a loaded model
- (Optional) Ollama with qwen3:1.7b
- (Optional) MEXC API credentials for live market data

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

# Optional: Pre-configured pipeline IDs (created by setup script)
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

Run only Agent 1 to scan markets without the full pipeline.

```bash
uv run python main.py scan
```

**Output:** Table of market signals with risk scores, directions, and regimes.

### Demo Mode (Video Recording)

Cinematic walkthrough with narration panels, step-by-step agent execution, and HITL simulation. Designed for a 3-4 minute screen recording.

```bash
uv run python main.py demo
```

**Includes:**
1. Problem/solution introduction
2. Scenario setup (JPY volatility spike)
3. Architecture overview
4. Infrastructure health check
5. Full pipeline execution with narration
6. HITL approval simulation
7. Summary with key metrics

### System Status

Check cluster health, node status, and performance metrics.

```bash
uv run python main.py status
```

### HITL Webhook Server

Start the FastAPI server for human-in-the-loop approval.

```bash
uv run python main.py hitl
```

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/pending` | List pending approvals |
| POST | `/approve` | Approve a strategy |
| POST | `/reject` | Reject and escalate |
| GET | `/report/{run_id}` | Get full run report |

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

### Windows Launcher

```bash
# Double-click or run from terminal
launchers\SENTINEL.bat
```

---

## Project Structure

```
aria-agents-hackathon-private/
+-- main.py                          # CLI entry point (5 modes)
+-- pyproject.toml                   # Dependencies (airia, ccxt, reportlab...)
+-- setup_airia_pipelines.py         # One-time Airia pipeline provisioning
+-- .env                             # API keys (gitignored)
+-- .env.example                     # Template
+-- src/
|   +-- __init__.py
|   +-- config.py                    # Centralized config (Airia, cluster, markets)
|   +-- models.py                    # Pydantic models (Signal, Exposure, Strategy...)
|   +-- database.py                  # SQLite audit trail (5 tables)
|   +-- orchestrator.py              # Main pipeline orchestration (3 phases)
|   +-- airia_bridge.py              # Airia SDK wrapper (pipeline + temporary mode)
|   +-- agents/
|   |   +-- __init__.py
|   |   +-- market_intelligence.py   # Agent 1: CCXT scan + Airia enrichment
|   |   +-- corporate_context.py     # Agent 2: Treasury analysis + Airia risks
|   |   +-- consensus_strategy.py    # Agent 3: 3-way multi-IA consensus
|   |   +-- compliance_docs.py       # Agent 4: PDF report + Airia summary
|   +-- services/
|   |   +-- market_data.py           # CCXT + simulated Forex/Commodities
|   |   +-- lm_cluster.py            # LM Studio + Ollama interface
|   |   +-- pdf_generator.py         # ReportLab PDF generation
|   |   +-- hitl_webhook.py          # FastAPI HITL server
|   +-- utils/
|       +-- http_pool.py             # Async HTTP connection pool (httpx)
|       +-- retry.py                 # Retry with exponential backoff
+-- data/
|   +-- mock_corporate.json          # Simulated ACME Corp treasury data
|   +-- sentinel.db                  # SQLite database (auto-created)
|   +-- reports/                     # Generated PDF reports
+-- demo/
|   +-- demo_scenario.py             # Cinematic demo script
|   +-- screenshots/                 # Screen captures
+-- launchers/
    +-- SENTINEL.bat                 # Windows launcher
```

---

## Airia Integration

### Dual Execution Modes

1. **Pipeline Mode** (recommended): Uses pre-configured pipeline IDs from `.env`
   - `execute_pipeline(pipeline_id, user_input)` via Airia SDK
   - Full traceability on the Airia platform

2. **Temporary Assistant Mode** (zero-setup): Uses embedded system prompts
   - `execute_temporary_assistant()` via Airia SDK
   - No pipeline creation needed — just an API key

The bridge automatically selects the right mode based on available pipeline IDs.

### Airia Infrastructure

| Resource | Count | Description |
|----------|-------|-------------|
| Pipelines | 4 | One per agent (Market, Corporate, Consensus, Compliance) |
| Deployments | 4 | Production deployments for each pipeline |
| Custom Tools | 3 | HITL Webhook, Market Scanner, Report Generator |
| Model | GPT-5.1 | Cloud model used for pipeline execution |

### System Prompts

Each agent has a carefully crafted system prompt in French, embedded in `airia_bridge.py`:
- **Agent 1**: Analyze market data, return JSON with risk scores and regimes
- **Agent 2**: Analyze treasury data, identify concentration risks, return JSON
- **Agent 3**: Generate 3 hedging strategies (Conservative/Moderate/Aggressive) in JSON
- **Agent 4**: Write formal executive summary in Big Four style

---

## Database Schema

SQLite with WAL mode, 5 tables:

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `signals` | Market scan results | symbol, price, risk_score, direction, regime |
| `exposures` | Corporate exposure | company_id, total_assets, risk_score, positions |
| `strategies` | Hedging strategies | name, confidence, cost, risk_reduction |
| `approvals` | HITL approvals | strategy_name, status, approved_by |
| `audit_log` | Full audit trail | step, agent, model_used, latency_ms |

All tables indexed by `run_id` for fast pipeline queries.

---

## Key Features

- **Multi-IA Consensus**: Queries 3 AI models in parallel (LM Studio + Ollama + Airia), averages confidence scores, detects dissenting views
- **Real Market Data**: Live crypto prices via CCXT (MEXC), simulated Forex/Commodities for demo
- **Risk Scoring**: Multi-factor scoring (volatility, momentum, regime classification)
- **Concentration Detection**: Identifies over-exposed currencies and unhedged commodity positions
- **3 Hedging Strategies**: Conservative, Moderate, Aggressive with cost/risk tradeoffs and rationale
- **Professional PDF Reports**: Executive summary, market analysis, exposure breakdown, strategy comparison table, audit trail
- **Airia Executive Summary**: Cloud-generated formal summary injected into PDF (Big Four style)
- **Full Audit Trail**: Every agent step logged to SQLite with timestamps, model info, and latency
- **HITL Approval**: FastAPI webhook for human review with approve/reject/escalate flow
- **Hybrid Architecture**: Works fully offline with local AI cluster; Airia enrichment optional
- **Cinematic Demo Mode**: Built-in demo script with narration for video recording

---

## Hackathon Criteria Mapping

| Criteria | Our Response |
|----------|-------------|
| **Technical Implementation** | Airia SDK (4 pipelines + deployments) + Local AI Cluster (5 GPU, 43GB VRAM) + CCXT + Multi-IA 3-way Consensus + FastAPI HITL |
| **UX/UI Design** | Rich CLI dashboard with tables + Professional PDF Reports (ReportLab) + HITL Webhook REST API |
| **Potential Impact** | Enterprise Treasury Management is a massive market ($1T+). AI-powered risk orchestration reduces decision latency from days to minutes |
| **Creativity/Uniqueness** | Fusion of local AI infrastructure (5 GPU cluster) + cloud Airia platform + multi-model consensus voting. No other solution combines local LLM inference with Airia pipelines |

---

## Performance (Typical Run)

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

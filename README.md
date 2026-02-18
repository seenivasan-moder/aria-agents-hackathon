# Airia Sentinel

**Multi-Agent Treasury Orchestration & Risk Management**

> Airia AI Agents Challenge - Track 2: Active Agents

## Problem

Enterprise treasury teams face a constant challenge: monitoring global markets (Forex, Crypto, Commodities), assessing internal financial exposure, and making timely hedging decisions. Today, this process is fragmented across multiple tools, slow, and prone to human error.

## Solution

Airia Sentinel is a **multi-agent AI system** that orchestrates the entire treasury risk management workflow:

1. **Market Intelligence Agent** scans crypto, forex, and commodity markets in real-time
2. **Corporate Context Agent** analyzes internal financial exposure and concentration risks
3. **Consensus & Strategy Agent** generates hedging strategies using multi-IA consensus (3 AI models in parallel)
4. **Compliance & Docs Agent** produces a professional PDF report with full audit trail

All orchestrated through **Airia pipelines** with **Human-in-the-Loop (HITL)** approval via webhook.

## Architecture

```
                    +---------------------+
                    |   AIRIA PLATFORM    |
                    |   (Orchestrator)    |
                    +---------+-----------+
                              |
            +-----------------+------------------+
            |                 |                  |
    +-------v-------+ +------v------+ +---------v-------+
    |  Agent 1      | |  Agent 2    | |  Agent 3        |
    |  Market       | |  Corporate  | |  Consensus      |
    |  Intelligence | |  Context    | |  & Strategy     |
    +-------+-------+ +------+------+ +---------+-------+
            |                |                   |
            +----------------+-------------------+
                             |
                    +--------v--------+
                    |   Agent 4       |
                    |   Compliance    |
                    |   & Docs        |
                    +--------+--------+
                             |
                    +--------v--------+
                    |   HITL          |
                    |   Webhook       |
                    |   Approval      |
                    +-----------------+
```

## Tech Stack

| Component | Technology |
|---|---|
| AI Orchestration | Airia SDK + Custom Pipeline |
| Local AI Cluster | LM Studio (qwen3-30b, 5 GPUs, 43GB VRAM) |
| Lightweight AI | Ollama (qwen3:1.7b) |
| Market Data | CCXT (MEXC, multi-exchange) |
| Data Models | Pydantic v2 |
| PDF Reports | ReportLab |
| HITL Webhook | FastAPI + Uvicorn |
| Database | SQLite (audit trail) |
| UI | Rich (terminal) |
| Package Manager | uv |
| Language | Python 3.13 |

## Quick Start

```bash
# Install dependencies
uv sync

# Configure API keys
cp .env.example .env
# Edit .env with your AIRIA_API_KEY, MEXC credentials, etc.

# Run full pipeline
uv run python main.py

# Market scan only
uv run python main.py scan

# Check cluster status
uv run python main.py status

# Start HITL approval server
uv run python main.py hitl

# Run demo scenario
uv run python main.py demo
```

## Key Features

- **Multi-IA Consensus**: Queries multiple AI models (LM Studio + Ollama) in parallel, averages confidence scores
- **Real Market Data**: Live crypto prices via CCXT, simulated Forex/Commodities
- **Risk Scoring**: Multi-factor scoring (volatility, momentum, regime classification)
- **Concentration Detection**: Identifies over-exposed currencies and unhedged commodities
- **3 Hedging Strategies**: Conservative, Moderate, Aggressive with cost/risk tradeoffs
- **Professional PDF Reports**: Executive summary, market analysis, exposure breakdown, strategy comparison
- **Full Audit Trail**: Every agent step logged to SQLite with timestamps and model info
- **HITL Approval**: FastAPI webhook for human review (approve/reject strategies)
- **Local-First**: Works fully offline with local AI cluster, Airia optional

## Project Structure

```
src/
  config.py              # Centralized configuration
  models.py              # Pydantic data models
  database.py            # SQLite audit trail
  orchestrator.py        # Main pipeline orchestration
  airia_bridge.py        # Airia SDK wrapper
  agents/
    market_intelligence.py  # Agent 1: market scanning
    corporate_context.py    # Agent 2: corporate analysis
    consensus_strategy.py   # Agent 3: multi-IA consensus
    compliance_docs.py      # Agent 4: reports + audit
  services/
    market_data.py       # CCXT + simulated data
    lm_cluster.py        # LM Studio/Ollama interface
    pdf_generator.py     # ReportLab PDF generation
    hitl_webhook.py      # FastAPI HITL server
  utils/
    http_pool.py         # Async HTTP connection pool
    retry.py             # Retry with exponential backoff
```

## Hackathon Criteria

| Criteria | Our Response |
|---|---|
| Technical Implementation | Airia SDK + Local AI Cluster (5 GPU) + CCXT + Multi-IA Consensus |
| UX/UI Design | Rich CLI + Professional PDF Reports + HITL Webhook |
| Potential Impact | Enterprise Treasury = massive market, AI-powered risk management |
| Creativity/Uniqueness | Fusion of local AI (43GB VRAM) + cloud Airia + multi-model consensus |

## License

MIT

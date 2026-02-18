# Architecture — Airia Sentinel

Technical deep dive into the multi-agent orchestration system.

---

## System Overview

```mermaid
graph TB
    subgraph "User Interface"
        CLI["CLI (Rich)"]
        DEMO["Demo Mode"]
        HITL_UI["HITL Webhook"]
    end

    subgraph "Orchestrator"
        ORCH["orchestrator.py"]
    end

    subgraph "Phase 1 — Parallel Scan"
        A1["Agent 1<br/>Market Intelligence"]
        A2["Agent 2<br/>Corporate Context"]
    end

    subgraph "Phase 2 — Consensus"
        A3["Agent 3<br/>Consensus & Strategy"]
    end

    subgraph "Phase 3 — Report"
        A4["Agent 4<br/>Compliance & Docs"]
    end

    subgraph "AI Infrastructure"
        M1["LM Studio M1<br/>qwen3-30b<br/>5 GPU / 43GB"]
        OL1["Ollama OL1<br/>qwen3:1.7b"]
        AIRIA["Airia Platform<br/>GPT-5.1"]
    end

    subgraph "Data Layer"
        CCXT["CCXT / MEXC<br/>Live crypto data"]
        MOCK["Mock Corporate<br/>ACME Corp data"]
        DB["SQLite<br/>5 tables / WAL"]
        PDF["PDF Reports<br/>ReportLab"]
    end

    CLI --> ORCH
    DEMO --> ORCH
    ORCH --> A1
    ORCH --> A2
    A1 --> A3
    A2 --> A3
    A3 --> A4
    A4 --> HITL_UI

    A1 --> CCXT
    A1 --> AIRIA
    A2 --> MOCK
    A2 --> AIRIA
    A3 --> M1
    A3 --> OL1
    A3 --> AIRIA
    A4 --> PDF
    A4 --> AIRIA

    A1 --> DB
    A2 --> DB
    A3 --> DB
    A4 --> DB
```

---

## Pipeline Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI as main.py
    participant Orch as orchestrator.py
    participant A1 as Agent 1: Market
    participant A2 as Agent 2: Corporate
    participant A3 as Agent 3: Consensus
    participant A4 as Agent 4: Compliance
    participant CCXT as CCXT/MEXC
    participant M1 as LM Studio
    participant OL1 as Ollama
    participant Airia as Airia Platform
    participant DB as SQLite
    participant HITL as HITL Webhook

    User->>CLI: python main.py pipeline
    CLI->>Orch: run_full_pipeline()
    Orch->>DB: init_db()

    Note over Orch,A2: Phase 1 — Parallel Execution
    par Agent 1 + Agent 2
        Orch->>A1: run(run_id)
        A1->>CCXT: fetch_tickers(10 pairs)
        CCXT-->>A1: Live prices
        A1->>A1: Risk scoring + regime detection
        A1->>Airia: execute_market_pipeline()
        Airia-->>A1: Enriched risk scores
        A1->>DB: save_signals()
        A1-->>Orch: MarketSignal[]
    and
        Orch->>A2: run(run_id)
        A2->>A2: Load mock_corporate.json
        A2->>A2: Calculate exposure + concentration
        A2->>Airia: execute_corporate_pipeline()
        Airia-->>A2: Risk zones + liquidity status
        A2->>DB: save_exposure()
        A2-->>Orch: CorporateExposure
    end

    Note over Orch,Airia: Phase 2 — 3-Way Consensus
    Orch->>A3: run(run_id, signals, exposure)
    par 3 AI Sources
        A3->>M1: POST /v1/chat/completions
        M1-->>A3: Strategy JSON (qwen3-30b)
    and
        A3->>OL1: POST /api/chat
        OL1-->>A3: Strategy JSON (qwen3:1.7b)
    end
    A3->>Airia: execute_consensus_pipeline()
    Airia-->>A3: Strategy JSON (GPT-5.1)
    A3->>A3: Merge confidence scores + detect dissent
    A3->>DB: save_strategies()
    A3-->>Orch: ConsensusResult

    Note over Orch,HITL: Phase 3 — Report + HITL
    Orch->>A4: run(run_id, signals, exposure, consensus)
    A4->>Airia: execute_compliance_pipeline()
    Airia-->>A4: Executive summary (Big Four style)
    A4->>A4: Generate PDF (ReportLab)
    A4->>DB: save_approval(pending)
    A4->>DB: save_audit()
    A4-->>Orch: SentinelReport

    Orch-->>CLI: Display results
    User->>HITL: POST /approve
    HITL->>DB: save_approval(approved)
```

---

## Data Models

```mermaid
classDiagram
    class MarketSignal {
        +str symbol
        +str asset_class
        +float price
        +float change_24h
        +float volatility
        +float risk_score [0-100]
        +Direction direction
        +Regime regime
        +datetime timestamp
    }

    class CorporateExposure {
        +str company_id
        +str base_currency
        +float total_assets
        +float net_cash
        +list~CurrencyPosition~ positions
        +list~str~ concentration_risks
        +float risk_score [0-100]
    }

    class CurrencyPosition {
        +str currency
        +float cash_balance
        +float receivables
        +float payables
        +float net_exposure
        +float exposure_pct
    }

    class HedgingStrategy {
        +str name
        +str description
        +list~str~ instruments
        +float cost_estimate_pct
        +float risk_reduction_pct
        +float confidence [0-100]
        +str rationale
        +RiskLevel risk_level
    }

    class ConsensusResult {
        +list~HedgingStrategy~ strategies
        +int recommended_index
        +float consensus_score
        +list~str~ dissenting_views
        +list~str~ models_used
    }

    class SentinelReport {
        +str run_id
        +str company_id
        +list~MarketSignal~ market_signals
        +CorporateExposure corporate_exposure
        +ConsensusResult consensus
        +list~AuditRecord~ audit_trail
        +str pdf_path
        +ApprovalStatus approval_status
    }

    class AuditRecord {
        +str run_id
        +str step
        +str agent
        +str model_used
        +float latency_ms
    }

    SentinelReport --> MarketSignal : contains
    SentinelReport --> CorporateExposure : contains
    SentinelReport --> ConsensusResult : contains
    SentinelReport --> AuditRecord : contains
    CorporateExposure --> CurrencyPosition : contains
    ConsensusResult --> HedgingStrategy : contains
```

---

## Airia Bridge — Dual Execution Mode

```mermaid
flowchart TD
    A[Agent calls bridge] --> B{Pipeline ID configured?}
    B -->|Yes| C[execute_pipeline]
    B -->|No| D[execute_temporary_assistant]

    C --> E[Airia SDK]
    D --> F[Embedded FR system prompt]
    F --> E

    E --> G{Response OK?}
    G -->|Yes| H[Parse JSON from response]
    G -->|No| I[Return error, agent uses local-only]

    H --> J[Return parsed result + metadata]

    style C fill:#4CAF50,color:#fff
    style D fill:#2196F3,color:#fff
    style I fill:#f44336,color:#fff
```

Each agent has a **French system prompt** embedded in `airia_bridge.py`:

| Agent | Prompt Focus | Output Format |
|-------|-------------|---------------|
| Sentinel-1 | Risk scoring, regime detection | JSON: `{signals: [...]}` |
| Sentinel-2 | Exposure analysis, concentration risks | JSON: `{net_exposure, risk_zones}` |
| Sentinel-3 | 3 hedging strategies with rationale | JSON: `{strategies: [...]}` |
| Sentinel-4 | Formal executive summary (Big Four style) | Structured text with sections |

---

## Consensus Algorithm

The 3-way consensus in Agent 3 works as follows:

```mermaid
flowchart TD
    P[Build prompt with market + exposure data] --> Q1[Query LM Studio M1]
    P --> Q2[Query Ollama OL1]
    P --> Q3[Query Airia Pipeline]

    Q1 --> R1[Parse strategies JSON]
    Q2 --> R2[Parse strategies JSON]
    Q3 --> R3[Parse strategies JSON]

    R1 --> M[Merge: weighted average of confidence scores]
    R2 --> M
    R3 --> M

    M --> D{Confidence spread > 20%?}
    D -->|Yes| DV[Flag dissenting views]
    D -->|No| OK[Consensus achieved]

    DV --> REC[Select highest-confidence strategy as recommended]
    OK --> REC

    REC --> OUT[ConsensusResult with strategies + score + dissent]
```

**Merge rules:**
1. Use first complete strategy set as base
2. Average confidence scores across all sources
3. Detect dissenting views when confidence spread exceeds 20%
4. Recommend strategy with highest average confidence

---

## Database Schema

```mermaid
erDiagram
    SIGNALS {
        int id PK
        text run_id FK
        text symbol
        text asset_class
        real price
        real change_24h
        real volatility
        real risk_score
        text direction
        text regime
        text raw_json
        real timestamp
    }

    EXPOSURES {
        int id PK
        text run_id FK
        text company_id
        real total_assets
        real net_cash
        real risk_score
        text positions_json
        text concentration_risks
        real timestamp
    }

    STRATEGIES {
        int id PK
        text run_id FK
        text name
        text description
        text risk_level
        real confidence
        real cost_estimate_pct
        real risk_reduction_pct
        int recommended
        text rationale
        real timestamp
    }

    APPROVALS {
        int id PK
        text run_id FK
        text strategy_name
        text status
        text approved_by
        text comment
        real timestamp
    }

    AUDIT_LOG {
        int id PK
        text run_id FK
        text step
        text agent
        text input_summary
        text output_summary
        text model_used
        real latency_ms
        real timestamp
    }

    SIGNALS }o--|| AUDIT_LOG : "run_id"
    EXPOSURES }o--|| AUDIT_LOG : "run_id"
    STRATEGIES }o--|| AUDIT_LOG : "run_id"
    APPROVALS }o--|| AUDIT_LOG : "run_id"
```

---

## Infrastructure

```mermaid
graph LR
    subgraph "Machine 1 — Primary (5 GPU, 43GB VRAM)"
        LMS["LM Studio<br/>127.0.0.1:1234<br/>qwen3-30b (permanent)"]
        OLL["Ollama<br/>127.0.0.1:11434<br/>qwen3:1.7b"]
    end

    subgraph "Machine 2 — Code (3 GPU, 24GB VRAM)"
        LMS2["LM Studio<br/>192.168.1.26:1234<br/>deepseek-coder-v2-lite"]
    end

    subgraph "Cloud"
        AIRIA_C["Airia Platform<br/>4 pipelines<br/>GPT-5.1"]
        MEXC["MEXC Exchange<br/>Live crypto data"]
    end

    subgraph "Application"
        APP["Airia Sentinel<br/>Python 3.13 + uv"]
        FAST["FastAPI<br/>HITL Webhook<br/>Port 8900"]
        SQLT["SQLite<br/>WAL mode"]
    end

    APP --> LMS
    APP --> OLL
    APP --> LMS2
    APP --> AIRIA_C
    APP --> MEXC
    APP --> FAST
    APP --> SQLT
```

> **Important**: Always use `127.0.0.1` instead of `localhost` to avoid IPv6 DNS resolution latency (~10s) on Windows.

---

## HTTP Connection Management

```mermaid
flowchart TD
    REQ[Agent HTTP request] --> POOL{Connection pool exists?}
    POOL -->|No| CREATE[Create httpx.AsyncClient<br/>20 max connections<br/>10 keepalive<br/>300s expiry]
    POOL -->|Yes| REUSE[Reuse existing client]

    CREATE --> SEND[Send request]
    REUSE --> SEND

    SEND --> RETRY{Response OK?}
    RETRY -->|Error| BACK[Exponential backoff<br/>0.5s * 2^attempt]
    RETRY -->|OK| TRACK[Track latency metrics]
    BACK --> SEND

    TRACK --> RET[Return response]
```

---

## Pipeline Engine — Composable Orchestration Patterns

The Pipeline Engine (`src/pipeline_engine.py`) provides 3 composable patterns:

```mermaid
graph TB
    subgraph "Pattern 1: Domino"
        D1["Step A"] --> DV1["Validator"]
        DV1 -->|pass| D2["Step B"]
        DV1 -->|fail| DSTOP["HALT"]
        D2 --> DV2["Validator"]
        DV2 -->|pass| D3["Step C"]
    end

    subgraph "Pattern 2: Vectorial"
        VA["Agent M1"] --> AGG["Aggregator"]
        VB["Agent OL1"] --> AGG
        VC["Agent Airia"] --> AGG
        AGG --> ENR["Enricher"]
    end

    subgraph "Pattern 3: Matrix"
        GRID["Score Grid<br/>Agent × Context"] --> TOP["Top-K Selection"]
        TOP --> EX1["Execute Best"]
        TOP --> EX2["Execute 2nd"]
        TOP --> EX3["Execute 3rd"]
    end

    subgraph "Composite"
        CP1["Vectorial"] --> CPV["Validator"]
        CPV --> CP2["Domino"]
        CP2 --> CP3["Matrix"]
    end
```

### Pipeline Engine Classes

```mermaid
classDiagram
    class StepResult {
        +str step_name
        +StepStatus status
        +Any data
        +float confidence
        +str agent_used
        +str model_used
        +float latency_ms
        +str error
    }

    class PipelineResult {
        +str pipeline_name
        +str pattern
        +list~StepResult~ steps
        +float total_latency_ms
        +float success_rate
        +Any final_data
    }

    class DominoPipeline {
        +str name
        +list~DominoStep~ steps
        +run(run_id) PipelineResult
    }

    class VectorialPipeline {
        +str name
        +list~VectorAgent~ agents
        +str aggregation
        +run(run_id, input_data) PipelineResult
    }

    class MatrixPipeline {
        +str name
        +list~MatrixCell~ cells
        +int top_k
        +run(run_id) PipelineResult
    }

    class CompositePipeline {
        +str name
        +list~Pipeline~ stages
        +run(run_id) PipelineResult
    }

    DominoPipeline --> StepResult
    VectorialPipeline --> StepResult
    MatrixPipeline --> StepResult
    CompositePipeline --> PipelineResult
```

### Matrix Agents

```mermaid
graph LR
    subgraph "Matrix Agent Group"
        VAL["Validator<br/>5 rule types<br/>error/warning severity"]
        AGG2["Aggregator<br/>4 fusion strategies<br/>dissent detection"]
        OPT["Optimizer<br/>Node profiling<br/>Efficiency scoring"]
        ENR2["Enricher<br/>AI summaries<br/>Metadata tags"]
    end

    subgraph "Integration Points"
        PE["Pipeline Engine"]
        DB2["SQLite Audit"]
        LM["LM Studio M1"]
        OL["Ollama OL1"]
        AI["Airia Platform"]
    end

    PE --> VAL
    PE --> AGG2
    PE --> OPT
    PE --> ENR2
    VAL --> DB2
    AGG2 --> DB2
    OPT --> DB2
    ENR2 --> LM
    ENR2 --> OL
    ENR2 --> AI
    ENR2 --> DB2
```

---

*See the main [README.md](../README.md) for quick start instructions and [USE_CASES.md](USE_CASES.md) for domain-specific applications.*

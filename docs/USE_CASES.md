# Use Cases — Airia Sentinel Multi-Agent Platform

Airia Sentinel's architecture is **domain-agnostic**. The core pattern — *parallel data collection → multi-model consensus → structured output + human approval* — applies across industries. Below are 4 production-ready use cases demonstrating the platform's versatility.

---

## 1. Treasury Risk Management (Current Implementation)

> **Status**: Fully implemented and evaluated on Airia Platform

### Problem
Enterprise treasury teams must monitor global markets (Forex, Crypto, Commodities) 24/7, assess internal financial exposure, and make hedging decisions under time pressure. The process is fragmented, slow, and error-prone.

### Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TREASURY RISK PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1 (Parallel)                                              │
│  ┌─────────────────────┐   ┌─────────────────────┐              │
│  │ Agent 1: Market      │   │ Agent 2: Corporate   │              │
│  │ Intelligence         │   │ Context              │              │
│  │                      │   │                      │              │
│  │ - CCXT (MEXC) live   │   │ - Treasury balance   │              │
│  │ - 10 pairs monitored │   │ - 5 currency pos.    │              │
│  │ - Risk scoring       │   │ - Concentration det. │              │
│  │ - Regime detection   │   │ - Commodity exposure  │              │
│  │ - Airia enrichment   │   │ - Airia enrichment   │              │
│  └──────────┬──────────┘   └──────────┬──────────┘              │
│             └───────────┬──────────────┘                         │
│                         ▼                                        │
│  Phase 2 (Sequential)                                            │
│  ┌──────────────────────────────────┐                            │
│  │ Agent 3: Consensus & Strategy     │                            │
│  │                                   │                            │
│  │ Source 1: LM Studio M1            │                            │
│  │   qwen3-30b (5 GPU, 43GB VRAM)   │                            │
│  │ Source 2: Ollama OL1              │                            │
│  │   qwen3:1.7b (fast inference)    │                            │
│  │ Source 3: Airia Pipeline          │                            │
│  │   GPT-5.1 (cloud)               │                            │
│  │                                   │                            │
│  │ → Weighted confidence averaging   │                            │
│  │ → Dissenting view detection       │                            │
│  │ → 3 strategies: Conserv/Mod/Aggr  │                            │
│  └──────────────┬────────────────────┘                            │
│                 ▼                                                 │
│  Phase 3 (Sequential)                                            │
│  ┌──────────────────────────────────┐                            │
│  │ Agent 4: Compliance & Docs        │                            │
│  │                                   │                            │
│  │ - Professional A4 PDF report      │                            │
│  │ - Airia executive summary         │                            │
│  │ - SQLite audit trail              │                            │
│  │ - HITL approval request           │                            │
│  └──────────────┬────────────────────┘                            │
│                 ▼                                                 │
│  ┌──────────────────────────────────┐                            │
│  │ HITL Gateway (FastAPI)            │                            │
│  │ CFO reviews → Approve / Reject    │                            │
│  └──────────────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### Key Metrics

| Metric | Value |
|--------|-------|
| Market pairs monitored | 10 (crypto + forex + commodities) |
| Currency positions analyzed | 5 |
| AI models in consensus | 3 (LM Studio + Ollama + Airia) |
| Pipeline latency | ~65 seconds end-to-end |
| Evaluation score | 30/30 passed |

### Business Value
- Reduces hedging decision latency from **days to minutes**
- Eliminates human error in risk assessment
- Full audit trail for regulatory compliance
- Multi-model consensus reduces single-model bias

---

## 2. AI Meta-Orchestrator — Multi-Provider Intelligence Manager

> **Status**: Architecture designed, ready for implementation on Airia

### Problem
Modern enterprises use **multiple AI providers simultaneously** (OpenAI, Anthropic, local LLMs, Ollama cloud models). This creates a management nightmare:
- No unified view of AI usage, costs, and quality
- No intelligent routing — queries go to the wrong model
- No quality control — hallucinations go undetected
- Token waste from redundant queries across providers

### Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 AI META-ORCHESTRATOR PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1 (Parallel)                                              │
│  ┌─────────────────────┐   ┌─────────────────────┐              │
│  │ Agent 1: Model       │   │ Agent 2: Quality     │              │
│  │ Router               │   │ Auditor              │              │
│  │                      │   │                      │              │
│  │ - Analyze query      │   │ - Compare responses  │              │
│  │   complexity         │   │   across providers   │              │
│  │ - Check model avail. │   │ - Detect hallucin.   │              │
│  │ - Estimate costs     │   │ - Score coherence    │              │
│  │ - Route to optimal   │   │ - Flag inconsist.    │              │
│  │   provider           │   │ - Benchmark quality  │              │
│  └──────────┬──────────┘   └──────────┬──────────┘              │
│             └───────────┬──────────────┘                         │
│                         ▼                                        │
│  Phase 2 (Sequential)                                            │
│  ┌──────────────────────────────────┐                            │
│  │ Agent 3: Cost Optimizer           │                            │
│  │                                   │                            │
│  │ - Aggregate usage across all      │                            │
│  │   providers (tokens, $, latency)  │                            │
│  │ - Identify waste patterns         │                            │
│  │ - Suggest model downgrades        │                            │
│  │   where quality is equivalent     │                            │
│  │ - Project monthly costs           │                            │
│  │ - ROI analysis per provider       │                            │
│  └──────────────┬────────────────────┘                            │
│                 ▼                                                 │
│  Phase 3 (Sequential)                                            │
│  ┌──────────────────────────────────┐                            │
│  │ Agent 4: Session Logger           │                            │
│  │                                   │                            │
│  │ - Log all AI interactions         │                            │
│  │ - Track metadata & provenance     │                            │
│  │ - Generate usage dashboards       │                            │
│  │ - Export compliance reports        │                            │
│  │ - HITL: budget approval alerts    │                            │
│  └──────────────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### Agents Description

| Agent | Input | Output | Airia Pipeline Role |
|-------|-------|--------|-------------------|
| **Model Router** | User query + context | Optimal provider selection + routing decision | Classify query complexity, predict cost |
| **Quality Auditor** | Multi-provider responses | Quality scores, hallucination flags, consistency report | Cross-validate responses, detect factual errors |
| **Cost Optimizer** | Usage logs + billing data | Cost projections, optimization recommendations | Analyze spending patterns, suggest downgrades |
| **Session Logger** | All pipeline metadata | Structured logs, compliance reports, dashboards | Generate executive usage summary |

### Supported Providers

| Provider | Type | Use Case |
|----------|------|----------|
| LM Studio (local) | Deep analysis | Complex reasoning, code generation |
| Ollama (local) | Fast inference | Quick queries, corrections |
| Ollama Cloud (minimax, glm, kimi) | Cloud + web search | Research, sub-agents |
| Airia Platform | Enterprise | Pipeline orchestration, evaluation |
| Claude / GPT (API) | Premium cloud | Critical decisions, creative work |

### Business Value
- **30-50% cost reduction** through intelligent routing
- Unified observability across all AI providers
- Automatic quality control and hallucination detection
- Full audit trail for enterprise AI governance

---

## 3. Smart File Organizer — AI-Powered Disk Intelligence

> **Status**: Architecture designed, ready for implementation on Airia

### Problem
Users and enterprises accumulate massive amounts of files across multiple drives. Manual organization is impractical:
- Duplicate files waste hundreds of GB
- Important documents buried in nested folders
- No automated classification or archiving policy
- Sensitive files (credentials, keys) scattered without protection

### Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  SMART FILE ORGANIZER PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1 (Parallel)                                              │
│  ┌─────────────────────┐   ┌─────────────────────┐              │
│  │ Agent 1: Scanner      │   │ Agent 2: Classifier  │              │
│  │                      │   │                      │              │
│  │ - Crawl target dirs  │   │ - Analyze file types │              │
│  │ - Hash for dedup     │   │ - AI content analysis│              │
│  │ - Size/date metadata │   │ - Sensitivity detect.│              │
│  │ - Extension mapping  │   │ - Category tagging   │              │
│  │ - Symlink resolution │   │ - Project grouping   │              │
│  └──────────┬──────────┘   └──────────┬──────────┘              │
│             └───────────┬──────────────┘                         │
│                         ▼                                        │
│  Phase 2 (Sequential)                                            │
│  ┌──────────────────────────────────┐                            │
│  │ Agent 3: Deduplicator             │                            │
│  │                                   │                            │
│  │ - Exact hash matching             │                            │
│  │ - Fuzzy content similarity        │                            │
│  │ - Suggest keep/delete decisions   │                            │
│  │ - Calculate recoverable space     │                            │
│  │ - Preserve latest version         │                            │
│  └──────────────┬────────────────────┘                            │
│                 ▼                                                 │
│  Phase 3 (Sequential)                                            │
│  ┌──────────────────────────────────┐                            │
│  │ Agent 4: Archiver & Reporter      │                            │
│  │                                   │                            │
│  │ - Generate organization plan      │                            │
│  │ - Create archive structure        │                            │
│  │ - Move/rename with undo log       │                            │
│  │ - PDF report (before/after)       │                            │
│  │ - HITL: confirm before deletion   │                            │
│  └──────────────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### Agents Description

| Agent | Input | Output | Airia Pipeline Role |
|-------|-------|--------|-------------------|
| **Scanner** | Directory paths, file system | File inventory with metadata, hashes | Index and structure raw scan data |
| **Classifier** | File metadata + content samples | Category labels, sensitivity flags, project groups | AI-powered content classification |
| **Deduplicator** | Hashed file inventory | Duplicate groups, keep/delete recommendations, space savings | Analyze similarity patterns |
| **Archiver** | Organization plan | Executed moves, archive structure, undo log, PDF report | Generate formal reorganization report |

### Business Value
- Recover **50-200GB+** of wasted disk space
- Automatic detection of sensitive files (credentials, API keys)
- AI-powered file classification (project, type, importance)
- Full undo capability with audit trail
- HITL approval before any destructive operation

---

## 4. JARVIS — Multi-Agent Personal AI Assistant

> **Status**: Partially implemented in the companion [JARVIS Turbo](https://github.com/Turbo31150) project

### Problem
Personal AI assistants are typically single-model, single-purpose tools. Users need an **orchestrated multi-agent system** that can:
- Understand voice commands in natural language
- Execute system operations (launch apps, manage files, run scripts)
- Perform deep research using multiple AI models
- Manage trading operations with real-time market data

### Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    JARVIS ASSISTANT PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────┐                            │
│  │ Voice Interface (STT/TTS)         │                            │
│  │ Whisper + Windows SAPI            │                            │
│  │ Wake word: "Jarvis"               │                            │
│  └──────────────┬────────────────────┘                            │
│                 ▼                                                 │
│  Phase 1 (Parallel)                                              │
│  ┌─────────────────────┐   ┌─────────────────────┐              │
│  │ Agent 1: Intent       │   │ Agent 2: Context     │              │
│  │ Classifier            │   │ Enricher             │              │
│  │                      │   │                      │              │
│  │ - Parse voice/text   │   │ - User history       │              │
│  │ - Detect intent      │   │ - Active apps/files  │              │
│  │ - Extract entities   │   │ - Time/location      │              │
│  │ - Route to pipeline  │   │ - Previous commands  │              │
│  └──────────┬──────────┘   └──────────┬──────────┘              │
│             └───────────┬──────────────┘                         │
│                         ▼                                        │
│  Phase 2 (Routed execution)                                      │
│  ┌──────────────────────────────────┐                            │
│  │ Agent 3: Execution Engine         │                            │
│  │                                   │                            │
│  │ Sub-agents (selected by intent):  │                            │
│  │ - ia-deep (Opus): Complex reason. │                            │
│  │ - ia-fast (Haiku): Quick answers  │                            │
│  │ - ia-system (Haiku): OS commands  │                            │
│  │ - ia-trading (Sonnet): Markets    │                            │
│  │ - ia-check (Sonnet): Validation   │                            │
│  │                                   │                            │
│  │ 83 MCP tools available:           │                            │
│  │ - Windows automation              │                            │
│  │ - Web browsing                    │                            │
│  │ - File operations                 │                            │
│  │ - LM Studio model management     │                            │
│  │ - Trading execution               │                            │
│  └──────────────┬────────────────────┘                            │
│                 ▼                                                 │
│  Phase 3 (Sequential)                                            │
│  ┌──────────────────────────────────┐                            │
│  │ Agent 4: Response & Memory        │                            │
│  │                                   │                            │
│  │ - Format response for voice/text  │                            │
│  │ - Update conversation memory      │                            │
│  │ - Log to database                 │                            │
│  │ - Suggest follow-up actions       │                            │
│  │ - TTS output via Windows SAPI     │                            │
│  └──────────────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### Capabilities

| Domain | Commands | Examples |
|--------|----------|---------|
| **System** | 438 registered commands | "Ouvre Chrome", "Lance VS Code", "Nettoie le bureau" |
| **AI Analysis** | Deep reasoning with local LLMs | "Analyse ce code", "Explique cette erreur" |
| **Trading** | MEXC Futures automation | "Scanne les marches", "Position sur BTC" |
| **Web** | Browser automation | "Cherche sur Google", "Ouvre YouTube" |
| **Voice** | Push-to-talk + wake word | "Jarvis" → "Je t'ecoute" |
| **Skills** | 77 registered pipelines | "Pipeline complet", "Status cluster" |

### Infrastructure

| Component | Specification |
|-----------|--------------|
| GPU Cluster | 5 GPUs, 43GB VRAM (RTX 2060 + 3x GTX 1660S + RTX 3080) |
| Primary Model | qwen3-30b (18.56GB, permanent on M1) |
| Fast Model | qwen3:1.7b on Ollama |
| Voice | Whisper STT + Windows SAPI TTS |
| MCP Tools | 83 tools across 6 categories |
| Database | SQLite (6 tables, full conversation history) |

### Business Value
- **Voice-first** personal AI assistant with local inference
- Zero cloud dependency for privacy-sensitive operations
- Multi-model routing for optimal cost/quality balance
- Extensible through MCP tools and Airia pipelines

---

## Cross-Cutting Architecture Pattern

All 4 use cases share the same **Airia Sentinel pattern**:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Phase 1    │     │   Phase 2    │     │   Phase 3    │     │    HITL      │
│  Parallel    │────>│  Consensus   │────>│  Report +    │────>│  Human       │
│  Data Scan   │     │  Multi-IA    │     │  Audit Trail │     │  Approval    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                    │                    │                    │
  2+ agents            3+ models            PDF + DB            FastAPI
  in parallel          in parallel          + Airia             webhook
```

This pattern is **reusable**, **testable** (via Airia evaluations), and **auditable** (via SQLite + HITL). It can be deployed across any domain where:
1. Multiple data sources need simultaneous scanning
2. Multiple AI models should reach consensus
3. Results require human validation before action
4. Full traceability is mandatory

---

*All use cases are designed to run on the Airia platform with the same SDK, evaluation suite, and deployment infrastructure.*

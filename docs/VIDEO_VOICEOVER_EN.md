# Airia Sentinel — Demo Video Voiceover Script (3 minutes)

**Target duration:** 2:50 - 3:00
**Tone:** Professional, confident, slightly excited
**Speed:** Moderate (not rushed). Pause between sections.

---

## [0:00 - 0:20] INTRO — The Problem

> "Enterprise treasury teams monitor dozens of financial instruments every single day — crypto, forex, commodities — while managing millions in corporate exposure. It's slow, fragmented, and error-prone."
>
> "What if sixteen specialized AI agents could do this in under ten seconds?"

*[Screen: Terminal launching, Airia Sentinel banner appears]*

---

## [0:20 - 0:40] WHAT IS AIRIA SENTINEL

> "This is Airia Sentinel — a multi-agent orchestration platform for treasury risk management. It runs a five-phase pipeline of sixteen AI agents, powered by Airia SDK, LM Studio, and Ollama."
>
> "Let me show you how it works."

*[Screen: Pipeline starts — Phase 1 banner appears]*

---

## [0:40 - 1:10] PHASE 1 & 2 — GATHER + ANALYZE

> "Phase one: six agents launch in parallel. Market Intelligence scans ten instruments in real-time via CCXT. Corporate Context analyzes internal treasury exposure. Meanwhile, the Anomaly Detector runs z-score analysis looking for volume spikes and contagion patterns. The Sentiment Scorer classifies market sentiment. The Liquidity Analyzer checks order book depth. And the News Scanner detects market-moving events."

*[Screen: Agents completing one by one, tables appearing]*

> "Phase two: three agents analyze the signals together. The Risk Aggregator fuses everything with weighted scoring and cascade detection. The Correlation Matrix identifies cross-asset dependencies. And the Volatility Forecaster computes Value-at-Risk estimates."

*[Screen: Risk Aggregator output, correlation table]*

---

## [1:10 - 1:40] PHASE 3 & 4 — STRATEGIZE + VALIDATE

> "Phase three: Consensus Strategy queries four different AI models in parallel — LM Studio M1, M2, Ollama, and Airia cloud — then merges their hedging recommendations using weighted voting. The Position Sizer calculates optimal allocation using Kelly Criterion and Risk Parity."

*[Screen: Consensus table with 3 strategies, position sizing]*

> "Phase four: validation. The Strategy Backtester runs a Monte Carlo simulation — fifty trajectories over thirty days — computing Sharpe ratio, max drawdown, and win rate. The Portfolio Optimizer computes the efficient frontier using Markowitz mean-variance optimization."

*[Screen: Backtester grades, optimizer weights]*

---

## [1:40 - 2:10] PHASE 5 — REPORT + HITL

> "Phase five: four agents run simultaneously. Compliance generates a PDF report enriched by Airia's executive summary. The Report Synthesizer assembles a six-section executive report. The Alert Agent dispatches warnings based on risk thresholds. And the Execution Strategist plans TWAP and VWAP order slicing with slippage estimation."

*[Screen: Phase 5 completing, PDF path shown]*

> "Now the Human-in-the-Loop gateway kicks in. No strategy executes without human approval. The CFO reviews the recommended hedge, sees the risk metrics, the backtest results, and clicks Approve."

*[Screen: HITL panel, approval simulation]*

---

## [2:10 - 2:35] DASHBOARD + ARCHITECTURE

> "Everything runs on a real-time web dashboard — you can see all sixteen agents completing live, market signals updating, risk gauges moving, the full audit trail, and cluster health for all five AI nodes."

*[Screen: Switch to browser — dashboard.html with WebSocket live updates]*

> "The architecture is fully hybrid: every agent tries Airia cloud first, then falls back to local GPU inference. This means the platform works even offline. All prompts are bilingual — English and French — with a single toggle."

---

## [2:35 - 3:00] CLOSING

> "To summarize: Airia Sentinel orchestrates sixteen AI agents across five phases, from market scanning to executive reporting, in under ten seconds. It combines Airia SDK with a local GPU cluster of nine GPUs and forty-six gigabytes of VRAM. Every decision is audited, every strategy is backtested, and every execution requires human approval."
>
> "This is Airia Sentinel — multi-agent treasury intelligence, powered by Airia."

*[Screen: Final summary panel — 16 agents, total time, "Pipeline v2 Done"]*

---

## Recording Tips

1. Launch `DEMO_ALL.bat` option 1 (pipeline v2) before recording
2. Have the dashboard open in Chrome at `http://127.0.0.1:8900/dashboard`
3. Split screen: terminal left, dashboard right
4. Record with OBS at 1080p minimum
5. Use a clear microphone, speak at moderate pace
6. Total video must be under 3 minutes (DevPost requirement)

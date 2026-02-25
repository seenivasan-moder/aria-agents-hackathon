"""Sentinel Orchestrator — runs the 4-agent and 16-agent pipelines end-to-end."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from rich.console import Console
from rich.panel import Panel

from src.config import config
from src.models import SentinelReport
from src import database as db
from src.airia_bridge import bridge
from src.agents import market_intelligence, corporate_context, consensus_strategy, compliance_docs
from src.agents.sentinel import anomaly_detector, sentiment_scorer, risk_aggregator, position_sizer
from src.agents.sentinel import backtester, report_synthesizer, alert_agent
from src.agents.sentinel import liquidity_analyzer, correlation_matrix, volatility_forecaster
from src.agents.sentinel import news_scanner, portfolio_optimizer, execution_strategist
from src.agents.meta import meta_router, context_manager, quality_auditor
from src.agents.organizer import librarian, dedup_agent
from src.agents.jarvis import intent_classifier, execution_engine

console = Console()


async def run_full_pipeline() -> SentinelReport:
    """Execute the complete Sentinel pipeline: scan -> analyze -> consensus -> report."""
    run_id = uuid.uuid4().hex[:12]
    t0 = time.monotonic()

    console.print(Panel(
        f"[bold white]AIRIA SENTINEL — Full Pipeline[/]\n"
        f"Run ID: {run_id}\n"
        f"Company: {config.company_id}\n"
        f"Airia: {'connected' if bridge.is_available else 'local-only'}",
        title="[bold cyan]Starting[/]",
        border_style="cyan",
    ))

    # Initialize database
    db.init_db()

    # Phase 1: Agents 1 + 2 in parallel (each agent handles Airia internally)
    console.print("\n[bold]Phase 1:[/] Market Intelligence + Corporate Context (parallel)")
    signals, exposure = await asyncio.gather(
        market_intelligence.run(run_id),
        corporate_context.run(run_id),
    )

    # Phase 2: Agent 3 — Consensus (3-way: M1 + OL1 + Airia)
    console.print("\n[bold]Phase 2:[/] Consensus & Strategy")
    consensus_result = await consensus_strategy.run(run_id, signals, exposure)

    # Phase 3: Agent 4 — Compliance & Report
    console.print("\n[bold]Phase 3:[/] Compliance & Documentation")
    report = await compliance_docs.run(run_id, signals, exposure, consensus_result)

    total_time = (time.monotonic() - t0) * 1000

    db.save_audit(
        run_id, "pipeline_complete", "orchestrator",
        output_summary=f"Total: {int(total_time)}ms, {len(signals)} signals, "
                       f"{len(consensus_result.strategies)} strategies, PDF generated",
        latency_ms=total_time,
    )

    console.print(Panel(
        f"[bold green]Pipeline Complete[/]\n"
        f"Run ID: {run_id}\n"
        f"Signals: {len(signals)}\n"
        f"Risk Score: {exposure.risk_score:.0f}/100\n"
        f"Strategies: {len(consensus_result.strategies)}\n"
        f"Recommended: {consensus_result.strategies[consensus_result.recommended_index].name if consensus_result.strategies else 'N/A'}\n"
        f"PDF: {report.pdf_path}\n"
        f"Total Time: {int(total_time)}ms\n"
        f"HITL: [yellow]Approval pending[/]",
        title="[bold green]Done[/]",
        border_style="green",
    ))

    return report


async def run_scan_only() -> list:
    """Run only the market scanning agent."""
    run_id = f"scan-{uuid.uuid4().hex[:8]}"
    db.init_db()
    console.print("[bold]Running market scan only...[/]")
    signals = await market_intelligence.run(run_id)
    return signals


async def run_full_pipeline_v2() -> SentinelReport:
    """Execute the enhanced 16-agent Sentinel pipeline.

    Phases:
        1 (parallel): Market Intelligence + Corporate Context, then
                       Anomaly Detector + Sentiment Scorer + Liquidity Analyzer + News Scanner
        2 (parallel): Risk Aggregator + Correlation Matrix + Volatility Forecaster
        3 (parallel): Consensus Strategy + Position Sizer
        4 (parallel): Strategy Backtester + Portfolio Optimizer
        5 (parallel): Compliance Docs + Report Synthesizer + Alert Agent + Execution Strategist
    """
    run_id = uuid.uuid4().hex[:12]
    t0 = time.monotonic()

    console.print(Panel(
        f"[bold white]AIRIA SENTINEL — Full Pipeline v2 (16 Agents)[/]\n"
        f"Run ID: {run_id}\n"
        f"Company: {config.company_id}\n"
        f"Airia: {'connected' if bridge.is_available else 'local-only'}",
        title="[bold cyan]Starting v2[/]",
        border_style="cyan",
    ))

    # Initialize database
    db.init_db()

    # ── Phase 1: GATHER (parallel) ──
    console.print("\n[bold]Phase 1/5:[/] Market Intelligence + Corporate Context + Anomaly + Sentiment + Liquidity + News (parallel)")

    # Market Intelligence and Corporate Context run first (they produce the raw signals)
    signals, exposure = await asyncio.gather(
        market_intelligence.run(run_id),
        corporate_context.run(run_id),
    )

    # Convert MarketSignal objects to dicts for sentinel agents
    signal_dicts = [{"symbol": s.symbol, "asset_class": s.asset_class, "price": s.price,
                     "change_1h": s.change_1h, "change_24h": s.change_24h, "change_7d": s.change_7d,
                     "volume_24h": s.volume_24h, "volatility": s.volatility, "risk_score": s.risk_score,
                     "direction": s.direction.value, "regime": s.regime.value} for s in signals]

    # Run 4 analysis agents in parallel on the signal dicts
    anomaly_result, sentiment_result, liquidity_result, news_result = await asyncio.gather(
        anomaly_detector.run(run_id, signal_dicts),
        sentiment_scorer.run(run_id, signal_dicts),
        liquidity_analyzer.run(run_id, signal_dicts),
        news_scanner.run(run_id, signal_dicts),
    )

    # ── Phase 2: ANALYZE (parallel) ──
    console.print("\n[bold]Phase 2/5:[/] Risk Aggregator + Correlation Matrix + Volatility Forecaster (parallel)")

    risk_signals = []
    for sd in signal_dicts:
        risk_signals.append({**sd, "source": "market_intelligence"})
    risk_signals.append({
        "source": "corporate_context",
        "risk_score": exposure.risk_score,
        "direction": "neutral",
        "symbol": "CORPORATE",
        "asset_class": "corporate",
    })

    risk_agg_result, corr_result, vol_result = await asyncio.gather(
        risk_aggregator.run(run_id, risk_signals),
        correlation_matrix.run(run_id, signal_dicts),
        volatility_forecaster.run(run_id, signal_dicts),
    )
    risk_profile_data = risk_agg_result.data

    # ── Phase 3: STRATEGIZE (parallel) ──
    console.print("\n[bold]Phase 3/5:[/] Consensus Strategy + Position Sizer (parallel)")

    consensus_result, position_result = await asyncio.gather(
        consensus_strategy.run(run_id, signals, exposure),
        position_sizer.run(run_id, risk_profile_data, account_balance=config.account_balance),
    )

    # ── Phase 4: VALIDATE (parallel) ──
    console.print("\n[bold]Phase 4/5:[/] Strategy Backtester + Portfolio Optimizer (parallel)")

    strategies_dicts = [
        s.model_dump() if hasattr(s, "model_dump") else s
        for s in consensus_result.strategies
    ]
    backtest_result, optimizer_result = await asyncio.gather(
        backtester.run(run_id, strategies_dicts),
        portfolio_optimizer.run(run_id, signal_dicts, current_positions=position_result.data),
    )

    # ── Phase 5: REPORT (parallel) ──
    console.print("\n[bold]Phase 5/5:[/] Compliance + Report + Alerts + Execution (parallel)")

    # Prepare all_results_dict for report_synthesizer
    all_results_dict = {
        "market_intelligence": {"signals_count": len(signals), "status": "success"},
        "corporate_context": {"risk_score": exposure.risk_score, "company": exposure.company_id, "status": "success"},
        "anomaly_detection": anomaly_result.data or {},
        "sentiment_scoring": sentiment_result.data or {},
        "liquidity_analysis": liquidity_result.data or {},
        "news_scanning": news_result.data or {},
        "risk_aggregation": risk_profile_data or {},
        "correlation_matrix": corr_result.data or {},
        "volatility_forecast": vol_result.data or {},
        "consensus_strategy": {
            "strategies_count": len(consensus_result.strategies),
            "recommended": consensus_result.strategies[consensus_result.recommended_index].name if consensus_result.strategies else "N/A",
            "consensus_score": consensus_result.consensus_score,
            "status": "success",
        },
        "position_sizing": position_result.data or {},
        "backtesting": backtest_result.data or {},
        "portfolio_optimization": optimizer_result.data or {},
    }

    anomaly_data = anomaly_result.data.get("anomalies", []) if anomaly_result.data else []
    positions_list = position_result.data.get("positions", []) if position_result.data else []

    report, synthesis_result, alert_result, exec_result = await asyncio.gather(
        compliance_docs.run(run_id, signals, exposure, consensus_result),
        report_synthesizer.run(run_id, all_results_dict),
        alert_agent.run(run_id, risk_profile_data, anomaly_data),
        execution_strategist.run(run_id, positions_list, signal_dicts),
    )

    total_time = (time.monotonic() - t0) * 1000

    db.save_audit(
        run_id, "pipeline_v2_complete", "orchestrator",
        output_summary=(
            f"Total: {int(total_time)}ms, {len(signals)} signals, "
            f"{len(consensus_result.strategies)} strategies, "
            f"risk={risk_profile_data.get('overall_risk', 0):.0f}, "
            f"16 agents, PDF generated"
        ),
        latency_ms=total_time,
    )

    # ── Summary panel showing ALL 16 agent results ──
    recommended_name = (
        consensus_result.strategies[consensus_result.recommended_index].name
        if consensus_result.strategies else "N/A"
    )
    overall_risk = risk_profile_data.get("overall_risk", 0) if risk_profile_data else 0
    alert_level = risk_profile_data.get("alert_level", "?") if risk_profile_data else "?"
    anomaly_rate = anomaly_result.data.get("anomaly_rate", 0) if anomaly_result.data else 0
    stress_index = anomaly_result.data.get("market_stress_index", 0) if anomaly_result.data else 0
    global_sentiment = sentiment_result.data.get("global_sentiment", 0) if sentiment_result.data else 0
    global_label = sentiment_result.data.get("global_label", "?") if sentiment_result.data else "?"
    total_allocated = position_result.data.get("total_allocated_pct", 0) if position_result.data else 0
    best_strategy = backtest_result.data.get("best_strategy", "?") if backtest_result.data else "?"
    synth_risk_level = synthesis_result.data.get("risk_level", "?") if synthesis_result.data else "?"
    total_alerts = alert_result.data.get("total_alerts", 0) if alert_result.data else 0
    critical_alerts = alert_result.data.get("critical_count", 0) if alert_result.data else 0
    liq_index = liquidity_result.data.get("overall_liquidity_index", 0) if liquidity_result.data else 0
    corr_regime = corr_result.data.get("regime", "?") if corr_result.data else "?"
    portfolio_var = vol_result.data.get("portfolio_var_95", 0) if vol_result.data else 0
    max_sharpe = optimizer_result.data.get("max_sharpe", {}).get("sharpe_ratio", 0) if optimizer_result.data else 0
    exec_cost = exec_result.data.get("total_cost_bps", 0) if exec_result.data else 0
    news_sentiment = news_result.data.get("aggregate_sentiment", 0) if news_result.data else 0

    console.print(Panel(
        f"[bold green]Pipeline v2 Complete — 16 Agents[/]\n"
        f"Run ID: {run_id}\n\n"
        f"[bold cyan]Phase 1 — GATHER[/]\n"
        f"  [bold]1. Market Intelligence:[/]   {len(signals)} signals scanned\n"
        f"  [bold]2. Corporate Context:[/]     Risk: {exposure.risk_score:.0f}/100\n"
        f"  [bold]3. Anomaly Detector:[/]      Rate: {anomaly_rate:.1f}%, Stress: {stress_index:.0f}\n"
        f"  [bold]4. Sentiment Scorer:[/]      Global: {global_sentiment:+.0f} ({global_label})\n"
        f"  [bold]5. Liquidity Analyzer:[/]    Index: {liq_index:.0f}/100\n"
        f"  [bold]6. News Scanner:[/]          Sentiment: {news_sentiment:+.0f}\n\n"
        f"[bold cyan]Phase 2 — ANALYZE[/]\n"
        f"  [bold]7. Risk Aggregator:[/]       Risk: {overall_risk:.0f}/100, Alert: {alert_level}\n"
        f"  [bold]8. Correlation Matrix:[/]    Regime: {corr_regime}\n"
        f"  [bold]9. Volatility Forecaster:[/] VaR 95%: {portfolio_var:.2%}\n\n"
        f"[bold cyan]Phase 3 — STRATEGIZE[/]\n"
        f"  [bold]10. Consensus Strategy:[/]   {len(consensus_result.strategies)} strategies, Best: {recommended_name}\n"
        f"  [bold]11. Position Sizer:[/]       Allocated: {total_allocated:.1f}%\n\n"
        f"[bold cyan]Phase 4 — VALIDATE[/]\n"
        f"  [bold]12. Strategy Backtester:[/]  Best: {best_strategy}\n"
        f"  [bold]13. Portfolio Optimizer:[/]   Max Sharpe: {max_sharpe:.2f}\n\n"
        f"[bold cyan]Phase 5 — REPORT[/]\n"
        f"  [bold]14. Compliance Docs:[/]      PDF: {report.pdf_path}\n"
        f"  [bold]15. Report Synthesizer:[/]   Risk level: {synth_risk_level}\n"
        f"  [bold]16. Alert Agent:[/]          {total_alerts} alerts ({critical_alerts} critical)\n"
        f"  [bold]17. Execution Strategist:[/] Cost: {exec_cost:.1f} bps\n\n"
        f"Total Time: {int(total_time)}ms\n"
        f"HITL: [yellow]Approval pending[/]",
        title="[bold green]Pipeline v2 Done — 16 Agents[/]",
        border_style="green",
    ))

    return report


def run_dashboard():
    """Start the HITL dashboard server with WebSocket support."""
    from src.services.dashboard_ws import start_dashboard_server
    start_dashboard_server()


async def run_meta_exchange(prompt: str = "Analyse ce code Python et explique les bonnes pratiques") -> dict:
    """Execute the Meta-Exchange pipeline: route + context + quality audit."""
    run_id = f"meta-{uuid.uuid4().hex[:8]}"
    t0 = time.monotonic()
    db.init_db()

    console.print(Panel(
        f"[bold white]META-EXCHANGE — Multi-AI Orchestration[/]\nRun ID: {run_id}",
        title="[bold magenta]Starting[/]", border_style="magenta",
    ))

    # Phase 1: Route the prompt to optimal provider
    console.print("\n[bold]Phase 1:[/] Intelligent Routing")
    routing = await meta_router.run(run_id, prompt)

    # Phase 2: Check context / dedup
    console.print("\n[bold]Phase 2:[/] Context Management")
    session_id = f"session-{run_id}"
    ctx_stats = await context_manager.run(run_id, session_id)

    # Phase 3: Quality audit (simulate multi-provider responses)
    console.print("\n[bold]Phase 3:[/] Quality Audit")
    from src.services.lm_cluster import query_lm, query_ollama
    responses = []
    lm_result = await query_lm(prompt, "M1")
    if lm_result.get("ok"):
        responses.append({"provider": "M1", "model": "qwen3-30b", "content": lm_result["content"]})
    ol_result = await query_ollama(prompt, "OL1")
    if ol_result.get("ok"):
        responses.append({"provider": "OL1", "model": "qwen3:1.7b", "content": ol_result["content"]})

    scores = []
    if responses:
        scores = await quality_auditor.run(run_id, prompt, responses)

    total_time = (time.monotonic() - t0) * 1000
    db.save_audit(run_id, "meta_complete", "orchestrator",
                  output_summary=f"Routed to {routing.provider}, {len(responses)} responses audited",
                  latency_ms=total_time)

    console.print(Panel(
        f"[bold green]Meta-Exchange Complete[/]\nRun ID: {run_id}\n"
        f"Provider: {routing.provider} ({routing.model})\n"
        f"Responses audited: {len(responses)}\n"
        f"Total Time: {int(total_time)}ms",
        title="[bold green]Done[/]", border_style="green",
    ))
    return {"routing": routing, "context": ctx_stats, "scores": scores}


async def run_organizer(target_dir: str = "") -> dict:
    """Execute the Organizer pipeline: scan + classify + deduplicate."""
    run_id = f"org-{uuid.uuid4().hex[:8]}"
    t0 = time.monotonic()
    db.init_db()

    if not target_dir:
        target_dir = str(config.reports_dir.parent.parent)  # project root

    console.print(Panel(
        f"[bold white]ORGANIZER — Intelligent File Management[/]\n"
        f"Run ID: {run_id}\nTarget: {target_dir}",
        title="[bold magenta]Starting[/]", border_style="magenta",
    ))

    # Phase 1: Scan and classify
    console.print("\n[bold]Phase 1:[/] File Scanning & Classification")
    files = await librarian.run(run_id, target_dir)

    # Phase 2: Deduplication
    console.print("\n[bold]Phase 2:[/] Deduplication Analysis")
    dedup_report = await dedup_agent.run(run_id, files)

    total_time = (time.monotonic() - t0) * 1000
    db.save_audit(run_id, "organizer_complete", "orchestrator",
                  output_summary=f"{len(files)} files scanned, {dedup_report.total_duplicates} duplicates, "
                                 f"{dedup_report.recoverable_bytes / 1_048_576:.1f} MB recoverable",
                  latency_ms=total_time)

    console.print(Panel(
        f"[bold green]Organizer Complete[/]\nRun ID: {run_id}\n"
        f"Files scanned: {len(files)}\n"
        f"Duplicates found: {dedup_report.total_duplicates}\n"
        f"Recoverable space: {dedup_report.recoverable_bytes / 1_048_576:.1f} MB\n"
        f"Total Time: {int(total_time)}ms",
        title="[bold green]Done[/]", border_style="green",
    ))
    return {"files": len(files), "dedup": dedup_report}


async def run_jarvis(user_input: str = "status du systeme") -> dict:
    """Execute the JARVIS pipeline: classify intent + route to execution engine."""
    run_id = f"jarvis-{uuid.uuid4().hex[:8]}"
    t0 = time.monotonic()
    db.init_db()

    console.print(Panel(
        f"[bold white]JARVIS — Personal AI Assistant[/]\n"
        f"Run ID: {run_id}\nInput: {user_input}",
        title="[bold magenta]Starting[/]", border_style="magenta",
    ))

    # Phase 1: Intent classification
    console.print("\n[bold]Phase 1:[/] Intent Classification")
    intent = await intent_classifier.run(run_id, user_input)

    # Phase 2: Execution
    console.print("\n[bold]Phase 2:[/] Execution Engine")
    result = await execution_engine.run(run_id, intent)

    total_time = (time.monotonic() - t0) * 1000
    db.save_audit(run_id, "jarvis_complete", "orchestrator",
                  output_summary=f"Intent: {intent.domain.value}/{intent.action.value}, "
                                 f"Agent: {result.agent_used}, Success: {result.success}",
                  latency_ms=total_time)

    console.print(Panel(
        f"[bold green]JARVIS Complete[/]\nRun ID: {run_id}\n"
        f"Intent: {intent.domain.value}/{intent.action.value}\n"
        f"Agent: {result.agent_used}\n"
        f"Response: {result.response[:200]}\n"
        f"Total Time: {int(total_time)}ms",
        title="[bold green]Done[/]", border_style="green",
    ))
    return {"intent": intent, "result": result}


async def run_status() -> dict:
    """Check cluster health and system status."""
    from src.services.lm_cluster import cluster_health
    from src.utils.http_pool import get_all_metrics

    health = await cluster_health()
    metrics = get_all_metrics()

    console.print(Panel(
        f"[bold]Cluster Status[/]\n"
        f"Nodes online: {health['online']}/{health['total']}\n"
        f"Airia: {'connected' if bridge.is_available else 'not configured'}\n"
        f"Database: {config.db_path}",
        title="[bold cyan]Sentinel Status[/]",
        border_style="cyan",
    ))

    for node in health["nodes"]:
        if node.get("online"):
            console.print(f"  [green]OK[/] {node['node']} — {node.get('latency_ms', '?')}ms, models: {', '.join(node.get('models', []))}")
        else:
            console.print(f"  [red]--[/] {node['node']} — offline")

    if metrics:
        console.print("\n[bold]Performance Metrics:[/]")
        for node, m in metrics.items():
            console.print(f"  {node}: avg={m['avg_ms']:.0f}ms, min={m['min_ms']:.0f}ms, max={m['max_ms']:.0f}ms ({m['samples']} samples)")

    return {"health": health, "metrics": metrics}

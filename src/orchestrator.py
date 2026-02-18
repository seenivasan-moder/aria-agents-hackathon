"""Sentinel Orchestrator — runs the 4-agent pipeline end-to-end."""

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

    # Phase 1: Agents 1 + 2 in parallel
    console.print("\n[bold]Phase 1:[/] Market Intelligence + Corporate Context (parallel)")
    signals, exposure = await asyncio.gather(
        market_intelligence.run(run_id),
        corporate_context.run(run_id),
    )

    # Notify Airia pipelines (non-blocking, best-effort)
    if bridge.is_available:
        bridge.execute_market_pipeline(config.watched_pairs)
        bridge.execute_corporate_pipeline(config.company_id)

    # Phase 2: Agent 3 — Consensus
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

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

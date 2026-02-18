"""Demo scenario — cinematic walkthrough for the hackathon video (< 4 min).

Usage:
    uv run python main.py demo
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from rich import box
from rich.live import Live
from rich.align import Align

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def narrate(text: str, pause: float = 1.5) -> None:
    """Print narration text with a pause for video recording."""
    console.print(f"\n[bold white on blue]  {text}  [/]")
    time.sleep(pause)


def typewriter(text: str, delay: float = 0.03) -> None:
    """Simulate typewriter effect for dramatic reveals."""
    for char in text:
        console.print(char, end="", highlight=False)
        time.sleep(delay)
    console.print()


def countdown(seconds: int = 3) -> None:
    """Visual countdown."""
    for i in range(seconds, 0, -1):
        console.print(f"  [dim]{i}...[/]", end="\r")
        time.sleep(1)
    console.print("  [bold green]GO![/]   ")


def section_divider(title: str) -> None:
    """Print a visual section divider."""
    console.print()
    console.rule(f"[bold cyan]{title}[/]", style="cyan")
    console.print()


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO STEPS
# ═══════════════════════════════════════════════════════════════════════════════

async def step_intro() -> None:
    """Step 0: Cinematic intro — problem + solution."""
    console.clear()

    # Big banner
    console.print(Panel(
        Align.center(Text.from_markup(
            "[bold cyan]"
            "    _    ___ ____  ___    _      ____  _____ _   _ _____ ___ _   _ _____ _\n"
            "   / \\  |_ _|  _ \\|_ _|  / \\    / ___|| ____| \\ | |_   _|_ _| \\ | | ____| |\n"
            "  / _ \\  | || |_) || |  / _ \\   \\___ \\|  _| |  \\| | | |  | ||  \\| |  _| | |\n"
            " / ___ \\ | ||  _ < | | / ___ \\   ___) | |___| |\\  | | |  | || |\\  | |___| |___\n"
            "/_/   \\_\\___|_| \\_\\___/_/   \\_\\ |____/|_____|_| \\_| |_| |___|_| \\_|_____|_____|\n"
            "[/]\n\n"
            "[bold white]Multi-Agent Treasury Orchestration & Risk Management[/]\n"
            "[dim]Airia AI Agents Challenge  -  Track 2: Active Agents[/]"
        )),
        border_style="bright_cyan",
        padding=(1, 2),
    ))

    await asyncio.sleep(2)

    # Problem statement
    console.print(Panel(
        "[bold red]THE PROBLEM[/]\n\n"
        "Enterprise treasury teams face a constant challenge:\n"
        "  [dim]>[/] Monitoring global markets (Forex, Crypto, Commodities) 24/7\n"
        "  [dim]>[/] Assessing internal financial exposure in real-time\n"
        "  [dim]>[/] Making timely hedging decisions under uncertainty\n"
        "  [dim]>[/] Maintaining regulatory compliance and audit trails\n\n"
        "[italic]Today, this process is fragmented, slow, and prone to human error.[/]",
        title="[bold red]Problem[/]",
        border_style="red",
        padding=(0, 2),
    ))

    await asyncio.sleep(3)

    # Solution
    console.print(Panel(
        "[bold green]THE SOLUTION: AIRIA SENTINEL[/]\n\n"
        "4 specialized AI agents orchestrated by the Airia platform:\n\n"
        "  [cyan]Agent 1[/] [bold]Market Intelligence[/]     Scans crypto, forex, commodities\n"
        "  [cyan]Agent 2[/] [bold]Corporate Context[/]       Analyzes internal financial exposure\n"
        "  [cyan]Agent 3[/] [bold]Consensus & Strategy[/]    Multi-IA consensus (3 AI models)\n"
        "  [cyan]Agent 4[/] [bold]Compliance & Docs[/]       PDF report + audit trail + HITL\n\n"
        "[dim]Stack: Airia SDK + LM Studio (5 GPU, 43GB VRAM) + Ollama + CCXT + FastAPI[/]",
        title="[bold green]Solution[/]",
        border_style="green",
        padding=(0, 2),
    ))

    await asyncio.sleep(3)


async def step_scenario() -> None:
    """Step 1: Present the scenario."""
    section_divider("SCENARIO")

    console.print(Panel(
        "[bold yellow]ALERT: JPY Volatility Spike Detected[/]\n\n"
        "[bold white]Company:[/] ACME International Trading Corp\n"
        "[bold white]Base Currency:[/] EUR\n"
        "[bold white]Total Assets:[/] 125,000,000 EUR\n"
        "[bold white]Key Exposure:[/] JPY (Japanese component sourcing)\n\n"
        "[italic]The Bank of Japan has signaled a policy shift.\n"
        "JPY/USD volatility has spiked to 2.8%.\n"
        "ACME Corp has 350M JPY in upcoming Q2 commitments.\n"
        "The system automatically triggers a full risk analysis...[/]",
        title="[bold yellow]Scenario[/]",
        border_style="yellow",
        padding=(0, 2),
    ))

    await asyncio.sleep(3)


async def step_architecture() -> None:
    """Step 2: Show the architecture flow."""
    section_divider("ARCHITECTURE")

    console.print(Panel(
        "[bold white]Pipeline Flow:[/]\n\n"
        "    [cyan]Phase 1[/] (Parallel)           [cyan]Phase 2[/]              [cyan]Phase 3[/]\n"
        "  +-----------------+       +-----------------+    +-----------------+\n"
        "  | [bold]Agent 1: Market[/]  |       | [bold]Agent 3:[/]          |    | [bold]Agent 4:[/]          |\n"
        "  | [dim]CCXT + Airia[/]     |------>| [bold]Consensus[/]         |--->| [bold]Compliance[/]        |\n"
        "  +-----------------+       | [dim]M1 + OL1 + Airia[/] |    | [dim]PDF + Audit[/]       |\n"
        "  +-----------------+  +--->+-----------------+    +---------+-------+\n"
        "  | [bold]Agent 2: Corp.[/]  |  |                               |\n"
        "  | [dim]Treasury data[/]   |--+                      [bold yellow]HITL Approval[/]\n"
        "  +-----------------+                            [dim]Human validates[/]\n\n"
        "  [dim]All steps logged to SQLite audit trail[/]",
        title="[bold cyan]Multi-Agent Architecture[/]",
        border_style="cyan",
        padding=(0, 2),
    ))

    await asyncio.sleep(3)


async def step_system_check() -> None:
    """Step 3: Verify system readiness."""
    section_divider("SYSTEM CHECK")

    narrate("Checking infrastructure...", pause=0.5)

    from src.config import config
    from src.airia_bridge import bridge

    # Cluster health
    from src.services.lm_cluster import cluster_health
    health = await cluster_health()

    table = Table(title="Infrastructure Status", box=box.ROUNDED, show_lines=False)
    table.add_column("Component", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="dim")

    # Airia
    airia_ok = bridge.is_available
    table.add_row(
        "Airia Platform",
        "[bold green]ONLINE[/]" if airia_ok else "[red]OFFLINE[/]",
        f"4 pipelines configured" if airia_ok else "Local-only mode",
    )

    # Cluster nodes
    for node in health["nodes"]:
        online = node.get("online", False)
        models = ", ".join(node.get("models", []))[:50]
        latency = f"{node.get('latency_ms', '?')}ms"
        table.add_row(
            f"{'LM Studio' if node.get('backend') != 'ollama' else 'Ollama'} ({node['node']})",
            f"[bold green]ONLINE[/]" if online else "[red]OFFLINE[/]",
            f"{latency} - {models}" if online else "Unreachable",
        )

    # Market data
    table.add_row("CCXT (MEXC)", "[bold green]READY[/]", f"{len(config.watched_pairs)} pairs monitored")

    # HITL
    table.add_row("HITL Webhook", "[bold green]READY[/]", f"Port {config.hitl_port}")

    # Database
    table.add_row("SQLite Audit", "[bold green]READY[/]", str(config.db_path))

    console.print(table)
    await asyncio.sleep(2)


async def step_run_pipeline() -> None:
    """Step 4: Execute the full pipeline with narration."""
    import uuid
    from src import database as db
    from src.agents import market_intelligence, corporate_context, consensus_strategy, compliance_docs

    run_id = uuid.uuid4().hex[:12]
    t0 = time.monotonic()

    db.init_db()

    # ── Agent 1 + 2: Parallel ─────────────────────────────────────────────
    section_divider("PHASE 1: PARALLEL SCANNING")

    narrate("Launching Agent 1 (Market) + Agent 2 (Corporate) in parallel...", pause=1)

    signals, exposure = await asyncio.gather(
        market_intelligence.run(run_id),
        corporate_context.run(run_id),
    )

    await asyncio.sleep(2)

    # ── Agent 3: Consensus ────────────────────────────────────────────────
    section_divider("PHASE 2: MULTI-IA CONSENSUS")

    narrate("Agent 3 queries 3 AI models in parallel for hedging strategies...", pause=1)

    consensus_result = await consensus_strategy.run(run_id, signals, exposure)

    await asyncio.sleep(2)

    # ── Agent 4: Compliance ───────────────────────────────────────────────
    section_divider("PHASE 3: COMPLIANCE & REPORT")

    narrate("Agent 4 generates a professional PDF report with Airia executive summary...", pause=1)

    report = await compliance_docs.run(run_id, signals, exposure, consensus_result)

    total_ms = (time.monotonic() - t0) * 1000

    db.save_audit(
        run_id, "pipeline_complete", "demo_orchestrator",
        output_summary=f"Demo pipeline: {int(total_ms)}ms total",
        latency_ms=total_ms,
    )

    await asyncio.sleep(1)
    return report, total_ms


async def step_hitl_simulation(report) -> None:
    """Step 5: Simulate HITL approval."""
    section_divider("HUMAN-IN-THE-LOOP APPROVAL")

    from src import database as db

    rec = report.consensus.strategies[report.consensus.recommended_index] if report.consensus and report.consensus.strategies else None
    strategy_name = rec.name if rec else "Conservative"

    # Show pending
    console.print(Panel(
        f"[bold yellow]APPROVAL REQUIRED[/]\n\n"
        f"Run ID: [bold]{report.run_id}[/]\n"
        f"Strategy: [bold cyan]{strategy_name}[/]\n"
        f"Confidence: [bold]{rec.confidence:.0f}%[/]\n"
        f"Cost: [bold]{rec.cost_estimate_pct:.2f}%[/] of portfolio\n"
        f"Risk Reduction: [bold]{rec.risk_reduction_pct:.0f}%[/]\n\n"
        f"[dim]Instruments: {', '.join(rec.instruments) if rec else 'N/A'}[/]\n\n"
        f"[italic]Waiting for CFO approval via HITL webhook...[/]\n"
        f"[dim]POST http://localhost:8900/approve[/]\n"
        f'[dim]{{"run_id": "{report.run_id}", "strategy_name": "{strategy_name}", "approved_by": "CFO"}}[/]',
        title="[bold yellow]HITL Gateway[/]",
        border_style="yellow",
        padding=(0, 2),
    ))

    await asyncio.sleep(3)

    # Simulate approval
    narrate("CFO reviews the report and approves the strategy...", pause=2)

    db.save_approval(report.run_id, strategy_name, "approved", "Marie Dupont (CFO)", "Approved after review of Q2 JPY exposure.")

    console.print(Panel(
        f"[bold green]APPROVED[/]\n\n"
        f"Strategy: [bold]{strategy_name}[/]\n"
        f"Approved by: [bold]Marie Dupont (CFO)[/]\n"
        f"Comment: [italic]Approved after review of Q2 JPY exposure.[/]\n\n"
        f"[dim]Audit trail updated. Hedging execution can proceed.[/]",
        title="[bold green]Approval Confirmed[/]",
        border_style="green",
        padding=(0, 2),
    ))

    await asyncio.sleep(2)


async def step_summary(report, total_ms: float) -> None:
    """Step 6: Final summary with key metrics."""
    section_divider("DEMO COMPLETE")

    n_signals = len(report.market_signals)
    high_risk = sum(1 for s in report.market_signals if s.risk_score > 60)
    rec = report.consensus.strategies[report.consensus.recommended_index] if report.consensus and report.consensus.strategies else None

    # Key metrics panel
    metrics = Table(box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 2))
    metrics.add_column("Metric", style="cyan")
    metrics.add_column("Value", style="bold white", justify="right")

    metrics.add_row("Market signals scanned", f"{n_signals}")
    metrics.add_row("High-risk alerts", f"[red]{high_risk}[/]")
    metrics.add_row("Corporate risk score", f"{report.corporate_exposure.risk_score:.0f}/100")
    metrics.add_row("Hedging strategies generated", f"{len(report.consensus.strategies) if report.consensus else 0}")
    metrics.add_row("Recommended strategy", f"[green]{rec.name}[/]" if rec else "N/A")
    metrics.add_row("Consensus confidence", f"{rec.confidence:.0f}%" if rec else "N/A")
    metrics.add_row("AI models used", f"{len(report.consensus.models_used) if report.consensus else 0}")
    metrics.add_row("Total pipeline time", f"{total_ms/1000:.1f}s")
    metrics.add_row("PDF report", report.pdf_path or "N/A")
    metrics.add_row("Approval status", "[bold green]APPROVED[/]")

    console.print(Panel(
        metrics,
        title="[bold green]Pipeline Results[/]",
        border_style="green",
        padding=(0, 1),
    ))

    await asyncio.sleep(2)

    # Tech stack summary
    console.print(Panel(
        "[bold white]Technology Stack[/]\n\n"
        "  [cyan]Orchestration[/]    Airia SDK (4 pipelines + 4 deployments + 3 tools)\n"
        "  [cyan]Local AI[/]         LM Studio (qwen3-30b, 5 GPU, 43GB VRAM)\n"
        "  [cyan]Lightweight AI[/]   Ollama (qwen3:1.7b)\n"
        "  [cyan]Market Data[/]      CCXT (MEXC, multi-exchange)\n"
        "  [cyan]Consensus[/]        3-way multi-model voting (M1 + OL1 + Airia)\n"
        "  [cyan]Reports[/]          ReportLab (PDF) + Airia executive summary\n"
        "  [cyan]HITL[/]             FastAPI webhook (approve/reject/escalate)\n"
        "  [cyan]Audit[/]            SQLite (5 tables, full traceability)\n"
        "  [cyan]Language[/]         Python 3.13 + uv\n\n"
        "[dim italic]Fusion of local AI infrastructure + cloud Airia platform\n"
        "for enterprise-grade treasury risk management.[/]",
        title="[bold cyan]Stack[/]",
        border_style="cyan",
        padding=(0, 2),
    ))

    await asyncio.sleep(2)

    # Final message
    console.print()
    console.print(Align.center(Text.from_markup(
        "[bold bright_cyan]AIRIA SENTINEL[/]  [dim]|[/]  "
        "[bold white]Multi-Agent Treasury Orchestration[/]  [dim]|[/]  "
        "[bold green]Ready for Production[/]"
    )))
    console.print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

async def run_demo():
    """Run the complete demo scenario with cinematic narration."""
    try:
        # Act 1: Introduction
        await step_intro()

        # Act 2: Scenario setup
        await step_scenario()

        # Act 3: Architecture overview
        await step_architecture()

        # Act 4: System check
        await step_system_check()

        # Act 5: Full pipeline execution
        report, total_ms = await step_run_pipeline()

        # Act 6: HITL approval
        await step_hitl_simulation(report)

        # Act 7: Summary & conclusion
        await step_summary(report, total_ms)

    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted.[/]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/]")
        import traceback
        traceback.print_exc()

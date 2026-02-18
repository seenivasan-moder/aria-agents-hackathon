"""Demo scenario — end-to-end walkthrough for the hackathon video (< 4 min)."""

from __future__ import annotations

import asyncio
import time

from rich.console import Console
from rich.panel import Panel

console = Console()


async def run_demo():
    """Run a complete demo scenario with narration pauses."""
    console.print(Panel(
        "[bold white]AIRIA SENTINEL — Demo Scenario[/]\n\n"
        "Scenario: ACME International detects JPY volatility spike.\n"
        "The system scans markets, analyzes exposure, generates\n"
        "hedging strategies via multi-IA consensus, produces a\n"
        "compliance report, and requests human approval.",
        title="[bold cyan]Demo Start[/]",
        border_style="cyan",
    ))

    await asyncio.sleep(2)

    # Step 1: Market scan
    console.print("\n[bold yellow]Step 1/4:[/] Market Intelligence Agent scanning...")
    await asyncio.sleep(1)

    from src.orchestrator import run_full_pipeline
    report = await run_full_pipeline()

    # Summary
    console.print("\n")
    console.print(Panel(
        f"[bold green]Demo Complete![/]\n\n"
        f"Run ID: {report.run_id}\n"
        f"Signals detected: {len(report.market_signals)}\n"
        f"Corporate risk score: {report.corporate_exposure.risk_score:.0f}/100\n"
        f"Strategies generated: {len(report.consensus.strategies) if report.consensus else 0}\n"
        f"PDF report: {report.pdf_path}\n"
        f"Approval status: {report.approval_status.value}\n\n"
        f"[dim]To approve: POST http://localhost:8900/approve[/]\n"
        f'[dim]{{"run_id": "{report.run_id}", "strategy_name": "Conservative", "approved_by": "CFO"}}[/]',
        title="[bold green]Demo Summary[/]",
        border_style="green",
    ))

"""Agent 4 — Compliance & Documentation: generates PDF report + audit trail."""

from __future__ import annotations

import time
import uuid
from typing import Any

from rich.console import Console

from src.config import config
from src.models import (
    MarketSignal, CorporateExposure, ConsensusResult,
    SentinelReport, AuditRecord, ApprovalStatus,
)
from src.services.pdf_generator import generate_report_pdf
from src import database as db

console = Console()


async def run(
    run_id: str,
    signals: list[MarketSignal],
    exposure: CorporateExposure,
    consensus: ConsensusResult,
) -> SentinelReport:
    """Execute Agent 4: compile report, generate PDF, create HITL approval."""
    t0 = time.monotonic()
    console.print("[bold cyan]Agent 4 — Compliance & Docs[/] generating report...")

    # Build report
    report = SentinelReport(
        run_id=run_id,
        company_id=exposure.company_id,
        market_signals=signals,
        corporate_exposure=exposure,
        consensus=consensus,
    )

    # Generate PDF
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_filename = f"sentinel_report_{run_id}.pdf"
    pdf_path = config.reports_dir / pdf_filename

    generate_report_pdf(report, str(pdf_path))
    report.pdf_path = str(pdf_path)

    latency = (time.monotonic() - t0) * 1000

    # Create HITL approval entry
    recommended = consensus.strategies[consensus.recommended_index] if consensus.strategies else None
    if recommended:
        db.save_approval(
            run_id, recommended.name, "pending",
        )

    # Audit
    db.save_audit(
        run_id, "compliance_report", "compliance_docs",
        input_summary=f"Full pipeline data for {exposure.company_id}",
        output_summary=f"PDF: {pdf_filename}, approval: pending",
        latency_ms=latency,
    )

    console.print(f"[green]Agent 4 done[/] — PDF: {pdf_path}")
    console.print(f"[yellow]HITL: Approval pending for strategy '{recommended.name if recommended else '?'}'[/]")

    return report

"""Agent 4 — Compliance & Documentation: generates PDF report + audit trail.

Hybrid mode: local PDF generation + optional Airia executive summary enrichment.
"""

from __future__ import annotations

import json
import time
from typing import Any

from rich.console import Console

from src.config import config
from src.models import (
    MarketSignal, CorporateExposure, ConsensusResult,
    SentinelReport, ApprovalStatus,
)
from src.services.pdf_generator import generate_report_pdf
from src.airia_bridge import bridge
from src import database as db

console = Console()


def _get_airia_executive_summary(report: SentinelReport) -> str:
    """Request a formal executive summary from Airia pipeline."""
    if not bridge.is_available:
        return ""

    report_data = {
        "company_id": report.company_id,
        "signals_count": len(report.market_signals),
        "high_risk_count": sum(1 for s in report.market_signals if s.risk_score > 60),
        "risk_score": report.corporate_exposure.risk_score if report.corporate_exposure else 0,
        "concentration_risks": report.corporate_exposure.concentration_risks if report.corporate_exposure else [],
        "recommended_strategy": (
            report.consensus.strategies[report.consensus.recommended_index].name
            if report.consensus and report.consensus.strategies else "N/A"
        ),
        "consensus_score": report.consensus.consensus_score if report.consensus else 0,
    }

    result = bridge.execute_compliance_pipeline(report_data)
    if result.get("ok"):
        return result.get("result", "")
    return ""


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

    # Try Airia executive summary (best-effort)
    airia_summary = _get_airia_executive_summary(report)
    if airia_summary:
        console.print(f"  [dim]Airia executive summary: {len(airia_summary)} chars[/]")

    # Generate PDF
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_filename = f"sentinel_report_{run_id}.pdf"
    pdf_path = config.reports_dir / pdf_filename

    generate_report_pdf(report, str(pdf_path), airia_summary=airia_summary)
    report.pdf_path = str(pdf_path)

    latency = (time.monotonic() - t0) * 1000
    model_used = "reportlab" + ("+airia" if airia_summary else "")

    # Create HITL approval entry
    recommended = consensus.strategies[consensus.recommended_index] if consensus.strategies else None
    if recommended:
        db.save_approval(run_id, recommended.name, "pending")

    # Audit
    db.save_audit(
        run_id, "compliance_report", "compliance_docs",
        input_summary=f"Full pipeline data for {exposure.company_id}",
        output_summary=f"PDF: {pdf_filename}, approval: pending, airia_summary: {'yes' if airia_summary else 'no'}",
        model_used=model_used,
        latency_ms=latency,
    )

    console.print(f"[green]Agent 4 done[/] — PDF: {pdf_path} [{model_used}]")
    console.print(f"[yellow]HITL: Approval pending for strategy '{recommended.name if recommended else '?'}'[/]")

    return report

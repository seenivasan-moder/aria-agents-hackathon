"""PDF report generator using ReportLab."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)

from src.models import SentinelReport


def generate_report_pdf(report: SentinelReport, output_path: str, airia_summary: str = "") -> str:
    """Generate a professional PDF report from SentinelReport data."""
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=20 * mm, bottomMargin=20 * mm,
        leftMargin=15 * mm, rightMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "SentinelTitle", parent=styles["Title"],
        fontSize=22, spaceAfter=10, textColor=colors.HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        "SectionHead", parent=styles["Heading2"],
        fontSize=14, spaceBefore=15, spaceAfter=8,
        textColor=colors.HexColor("#16213e"),
    ))
    styles.add(ParagraphStyle(
        "SubInfo", parent=styles["Normal"],
        fontSize=9, textColor=colors.grey,
    ))

    elements = []

    # ── Header ──
    elements.append(Paragraph("AIRIA SENTINEL", styles["SentinelTitle"]))
    elements.append(Paragraph("Treasury Risk Assessment Report", styles["Heading3"]))
    elements.append(Spacer(1, 5 * mm))
    elements.append(Paragraph(
        f"Run ID: {report.run_id} | Company: {report.company_id} | "
        f"Date: {report.created_at.strftime('%Y-%m-%d %H:%M UTC')}",
        styles["SubInfo"],
    ))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a1a2e")))
    elements.append(Spacer(1, 8 * mm))

    # ── Executive Summary ──
    elements.append(Paragraph("1. Executive Summary", styles["SectionHead"]))

    n_signals = len(report.market_signals)
    high_risk = [s for s in report.market_signals if s.risk_score > 60]
    rec = report.consensus.strategies[report.consensus.recommended_index] if report.consensus and report.consensus.strategies else None

    summary_text = (
        f"This report analyzes {n_signals} market signals across crypto, forex, and commodities. "
        f"<b>{len(high_risk)} high-risk signals</b> were identified. "
    )
    if report.corporate_exposure:
        summary_text += (
            f"Corporate exposure analysis for {report.corporate_exposure.company_id} reveals "
            f"{len(report.corporate_exposure.concentration_risks)} concentration risks "
            f"with an overall risk score of {report.corporate_exposure.risk_score:.0f}/100. "
        )
    if rec:
        summary_text += (
            f"The recommended hedging strategy is <b>{rec.name}</b> "
            f"(confidence: {rec.confidence:.0f}%, cost: {rec.cost_estimate_pct:.2f}%)."
        )

    elements.append(Paragraph(summary_text, styles["Normal"]))

    # Airia-generated executive summary (when available)
    if airia_summary:
        elements.append(Spacer(1, 4 * mm))
        elements.append(Paragraph("<i>AI-Generated Executive Brief (via Airia Pipeline):</i>", styles["SubInfo"]))
        elements.append(Spacer(1, 2 * mm))
        # Split by sections if present
        for line in airia_summary.split("\n"):
            line = line.strip()
            if line:
                if line.startswith("[") and line.endswith("]"):
                    elements.append(Paragraph(f"<b>{line}</b>", styles["Normal"]))
                else:
                    elements.append(Paragraph(line, styles["Normal"]))

    elements.append(Spacer(1, 5 * mm))

    # ── Market Analysis ──
    elements.append(Paragraph("2. Market Analysis", styles["SectionHead"]))

    signal_data = [["Symbol", "Class", "Price", "24h %", "Volatility", "Risk"]]
    for s in report.market_signals[:12]:
        signal_data.append([
            s.symbol, s.asset_class,
            f"{s.price:.4f}" if s.price < 10 else f"{s.price:.2f}",
            f"{s.change_24h:+.2f}%",
            f"{s.volatility:.2f}%",
            f"{s.risk_score:.0f}",
        ])

    signal_table = Table(signal_data, colWidths=[80, 55, 70, 55, 60, 40])
    signal_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
    ]))
    elements.append(signal_table)
    elements.append(Spacer(1, 5 * mm))

    # ── Corporate Exposure ──
    if report.corporate_exposure:
        elements.append(Paragraph("3. Corporate Exposure", styles["SectionHead"]))

        exp = report.corporate_exposure
        elements.append(Paragraph(
            f"Total Assets: <b>{exp.total_assets:,.0f} {exp.base_currency}</b> | "
            f"Net Cash: <b>{exp.net_cash:,.0f} {exp.base_currency}</b> | "
            f"Risk Score: <b>{exp.risk_score:.0f}/100</b>",
            styles["Normal"],
        ))
        elements.append(Spacer(1, 3 * mm))

        exp_data = [["Currency", "Cash", "Receivables", "Payables", "Net Exposure", "% Total"]]
        for p in exp.positions:
            exp_data.append([
                p.currency,
                f"{p.cash_balance:,.0f}",
                f"{p.receivables:,.0f}",
                f"{p.payables:,.0f}",
                f"{p.net_exposure:+,.0f}",
                f"{p.exposure_pct:.1f}%",
            ])

        exp_table = Table(exp_data, colWidths=[50, 70, 70, 70, 75, 45])
        exp_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
        ]))
        elements.append(exp_table)

        if exp.concentration_risks:
            elements.append(Spacer(1, 3 * mm))
            elements.append(Paragraph("<b>Concentration Risks:</b>", styles["Normal"]))
            for risk in exp.concentration_risks:
                elements.append(Paragraph(f"  - {risk}", styles["Normal"]))

    elements.append(Spacer(1, 5 * mm))

    # ── Hedging Strategies ──
    if report.consensus and report.consensus.strategies:
        elements.append(Paragraph("4. Hedging Strategies", styles["SectionHead"]))

        strat_data = [["Strategy", "Risk Level", "Cost %", "Risk Red. %", "Confidence", "Rec."]]
        for i, s in enumerate(report.consensus.strategies):
            strat_data.append([
                s.name,
                s.risk_level.value.upper(),
                f"{s.cost_estimate_pct:.2f}%",
                f"{s.risk_reduction_pct:.0f}%",
                f"{s.confidence:.0f}%",
                "*" if i == report.consensus.recommended_index else "",
            ])

        strat_table = Table(strat_data, colWidths=[80, 60, 50, 65, 60, 30])
        strat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (2, 1), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
        ]))
        elements.append(strat_table)
        elements.append(Spacer(1, 3 * mm))

        # Strategy details
        for s in report.consensus.strategies:
            elements.append(Paragraph(f"<b>{s.name}:</b> {s.description}", styles["Normal"]))
            if s.instruments:
                elements.append(Paragraph(f"  Instruments: {', '.join(s.instruments)}", styles["Normal"]))
            elements.append(Paragraph(f"  Rationale: {s.rationale}", styles["Normal"]))
            elements.append(Spacer(1, 2 * mm))

        elements.append(Paragraph(
            f"Consensus Score: <b>{report.consensus.consensus_score:.1f}</b> | "
            f"Models: {', '.join(report.consensus.models_used)}",
            styles["SubInfo"],
        ))

    elements.append(Spacer(1, 8 * mm))

    # ── Audit Trail ──
    elements.append(Paragraph("5. Audit Trail", styles["SectionHead"]))
    elements.append(Paragraph(
        f"Run ID: {report.run_id} | Generated: {report.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')} | "
        f"Status: {report.approval_status.value.upper()}",
        styles["Normal"],
    ))

    elements.append(Spacer(1, 10 * mm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    elements.append(Paragraph(
        "Airia Sentinel v1.0 — Multi-Agent Treasury Orchestration. "
        "This report was generated by an AI-powered pipeline and requires human approval.",
        styles["SubInfo"],
    ))

    # Build PDF
    doc.build(elements)
    return output_path

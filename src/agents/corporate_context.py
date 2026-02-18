"""Agent 2 — Corporate Context: analyzes internal financial exposure.

Uses simulated corporate data (mock_corporate.json) for the demo.
"""

from __future__ import annotations

import json
import time
from typing import Any

from rich.console import Console
from rich.table import Table

from src.config import config
from src.models import CorporateExposure, CurrencyPosition
from src import database as db

console = Console()


def _load_mock_data() -> dict:
    with open(config.mock_data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _calculate_exposure(data: dict) -> CorporateExposure:
    """Calculate net exposure per currency and detect concentration risks."""
    treasury = data.get("treasury", {})
    positions = []
    total_net = 0

    for pos in data.get("currency_positions", []):
        net = pos["cash_balance"] + pos["receivables"] - pos["payables"]
        total_net += abs(net)
        positions.append(CurrencyPosition(
            currency=pos["currency"],
            cash_balance=pos["cash_balance"],
            receivables=pos["receivables"],
            payables=pos["payables"],
            net_exposure=net,
        ))

    # Calculate exposure percentages and detect concentration
    concentration_risks = []
    max_exposure_pct = data.get("risk_policies", {}).get("max_single_currency_exposure_pct", 25)

    for p in positions:
        p.exposure_pct = round(abs(p.net_exposure) / total_net * 100, 1) if total_net > 0 else 0
        if p.exposure_pct > max_exposure_pct:
            concentration_risks.append(
                f"{p.currency}: {p.exposure_pct:.1f}% exposure exceeds {max_exposure_pct}% limit"
            )

    # Add unhedged commodity risks
    for comm in data.get("commodity_exposures", []):
        if comm.get("hedged_months", 0) == 0:
            concentration_risks.append(
                f"{comm['commodity']}: UNHEDGED — ${comm['monthly_consumption_usd']:,.0f}/month"
            )

    # Overall risk score
    risk_score = min(100, len(concentration_risks) * 20 + sum(
        10 for p in positions if abs(p.net_exposure) > 5_000_000
    ))

    return CorporateExposure(
        company_id=data.get("company_id", config.company_id),
        base_currency=data.get("base_currency", config.base_currency),
        total_assets=treasury.get("total_assets", 0),
        total_liabilities=treasury.get("total_liabilities", 0),
        net_cash=treasury.get("cash_equivalents", 0),
        positions=positions,
        concentration_risks=concentration_risks,
        risk_score=risk_score,
    )


async def run(run_id: str) -> CorporateExposure:
    """Execute Agent 2: analyze corporate financial context."""
    t0 = time.monotonic()
    console.print("[bold cyan]Agent 2 — Corporate Context[/] analyzing...")

    data = _load_mock_data()
    exposure = _calculate_exposure(data)

    latency = (time.monotonic() - t0) * 1000

    # Save to DB
    db.save_exposure(run_id, exposure.model_dump(mode="json"))
    db.save_audit(
        run_id, "corporate_analysis", "corporate_context",
        input_summary=f"Company: {exposure.company_id}",
        output_summary=f"{len(exposure.positions)} currencies, {len(exposure.concentration_risks)} risks, score={exposure.risk_score}",
        latency_ms=latency,
    )

    # Display
    _print_exposure(exposure)

    console.print(f"[green]Agent 2 done[/] — risk score: {exposure.risk_score} in {int(latency)}ms")
    return exposure


def _print_exposure(exposure: CorporateExposure) -> None:
    table = Table(title=f"Corporate Exposure — {exposure.company_id}", show_lines=False)
    table.add_column("Currency", style="cyan")
    table.add_column("Cash", justify="right")
    table.add_column("Receivables", justify="right")
    table.add_column("Payables", justify="right")
    table.add_column("Net Exposure", justify="right")
    table.add_column("% Total", justify="right")

    for p in exposure.positions:
        net_color = "red" if p.net_exposure < 0 else "green"
        table.add_row(
            p.currency,
            f"{p.cash_balance:,.0f}",
            f"{p.receivables:,.0f}",
            f"{p.payables:,.0f}",
            f"[{net_color}]{p.net_exposure:+,.0f}[/]",
            f"{p.exposure_pct:.1f}%",
        )

    console.print(table)

    if exposure.concentration_risks:
        console.print("\n[bold red]Concentration Risks:[/]")
        for risk in exposure.concentration_risks:
            console.print(f"  [red]![/] {risk}")

    console.print(f"\nTotal Assets: {exposure.total_assets:,.0f} {exposure.base_currency}")
    console.print(f"Net Cash: {exposure.net_cash:,.0f} {exposure.base_currency}")

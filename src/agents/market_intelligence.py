"""Agent 1 — Market Intelligence: scans markets for risk signals.

Hybrid mode: local CCXT scan + optional Airia pipeline enrichment.
"""

from __future__ import annotations

import json
import time
from typing import Any

from rich.console import Console
from rich.table import Table

from src.config import config
from src.models import MarketSignal, Direction, Regime
from src.services.market_data import fetch_all_market_data
from src.airia_bridge import bridge
from src import database as db

console = Console()


def _classify_risk(signal: dict) -> float:
    """Multi-factor risk scoring: volatility, momentum, regime."""
    vol = abs(signal.get("volatility", 0))
    change = abs(signal.get("change_24h", 0))
    score = min(100, vol * 15 + change * 5)
    return round(score, 1)


def _build_signal(raw: dict) -> MarketSignal:
    return MarketSignal(
        symbol=raw["symbol"],
        asset_class=raw.get("asset_class", "crypto"),
        price=raw.get("price", 0),
        change_1h=raw.get("change_1h", 0),
        change_24h=raw.get("change_24h", 0),
        change_7d=raw.get("change_7d", 0),
        volume_24h=raw.get("volume_24h", 0),
        volatility=raw.get("volatility", 0),
        funding_rate=raw.get("funding_rate", 0),
        risk_score=_classify_risk(raw),
        direction=Direction(raw.get("direction", "neutral")),
        regime=Regime(raw.get("regime", "neutral")),
    )


def _enrich_with_airia(signals: list[MarketSignal]) -> list[MarketSignal]:
    """Optionally enrich signals with Airia pipeline analysis."""
    if not bridge.is_available:
        return signals

    market_data = [s.model_dump(mode="json") for s in signals[:15]]
    result = bridge.execute_market_pipeline(market_data)

    if not result.get("ok") or not result.get("parsed"):
        return signals

    parsed = result["parsed"]
    airia_signals = parsed.get("signals", []) if isinstance(parsed, dict) else []
    if not airia_signals:
        return signals

    # Merge Airia risk scores with local signals (average)
    airia_map = {s.get("asset", ""): s for s in airia_signals}
    for signal in signals:
        airia = airia_map.get(signal.symbol)
        if airia and isinstance(airia.get("risk_score"), (int, float)):
            # Weighted average: 60% local, 40% Airia
            signal.risk_score = round(signal.risk_score * 0.6 + airia["risk_score"] * 0.4, 1)

    console.print(f"  [dim]Airia enrichment: {len(airia_signals)} signals merged[/]")
    return signals


async def run(run_id: str) -> list[MarketSignal]:
    """Execute Agent 1: scan all markets and return risk signals."""
    t0 = time.monotonic()
    console.print("[bold cyan]Agent 1 — Market Intelligence[/] scanning...")

    raw_data = await fetch_all_market_data()
    signals = [_build_signal(r) for r in raw_data if r.get("symbol") != "ERROR"]

    # Enrich with Airia (best-effort)
    signals = _enrich_with_airia(signals)

    # Sort by risk score descending
    signals.sort(key=lambda s: s.risk_score, reverse=True)

    latency = (time.monotonic() - t0) * 1000
    model_used = "CCXT+local" + ("+airia" if bridge.is_available else "")

    # Save to DB
    db.save_signals(run_id, [s.model_dump(mode="json") for s in signals])
    db.save_audit(
        run_id, "market_scan", "market_intelligence",
        input_summary=f"{len(config.watched_pairs)} pairs watched",
        output_summary=f"{len(signals)} signals, top risk: {signals[0].symbol if signals else 'none'} ({signals[0].risk_score if signals else 0})",
        model_used=model_used,
        latency_ms=latency,
    )

    _print_signals(signals)
    console.print(f"[green]Agent 1 done[/] — {len(signals)} signals in {int(latency)}ms [{model_used}]")
    return signals


def _print_signals(signals: list[MarketSignal]) -> None:
    table = Table(title="Market Signals", show_lines=False)
    table.add_column("Symbol", style="cyan")
    table.add_column("Class", style="dim")
    table.add_column("Price", justify="right")
    table.add_column("24h", justify="right")
    table.add_column("Vol", justify="right")
    table.add_column("Risk", justify="right")
    table.add_column("Direction")
    table.add_column("Regime")

    for s in signals[:15]:
        color_24h = "red" if s.change_24h < 0 else "green"
        risk_color = "red" if s.risk_score > 60 else ("yellow" if s.risk_score > 30 else "green")
        table.add_row(
            s.symbol, s.asset_class,
            f"{s.price:.4f}" if s.price < 10 else f"{s.price:.2f}",
            f"[{color_24h}]{s.change_24h:+.2f}%[/]",
            f"{s.volatility:.2f}%",
            f"[{risk_color}]{s.risk_score:.0f}[/]",
            s.direction.value,
            s.regime.value,
        )

    console.print(table)

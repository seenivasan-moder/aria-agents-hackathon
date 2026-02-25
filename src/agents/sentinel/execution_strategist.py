"""Agent Sentinel — Execution Strategist: TWAP/VWAP order execution planning.

Implements:
- TWAP (Time-Weighted Average Price) execution splitting
- VWAP (Volume-Weighted Average Price) execution profiles
- Market impact estimation (sqrt model)
- Slippage estimation based on position/volume ratio
- Execution urgency scoring
- Cost analysis (spread + slippage + impact)

Uses numpy for calculations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.pipeline_engine import StepResult, StepStatus
from src.config import config
from src import database as db

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

IMPACT_CONSTANT = 0.1       # Market impact scaling constant
MAX_PARTICIPATION = 0.05    # Max 5% of daily volume per slice
TWAP_DEFAULT_SLICES = 12    # Default TWAP slices (5-min intervals = 1 hour)
VWAP_VOLUME_PROFILE = [     # Typical 24h volume distribution (hourly %)
    2.5, 2.0, 1.5, 1.5, 2.0, 3.0,   # 00-05 (low)
    5.0, 7.0, 8.0, 7.0, 6.0, 5.5,   # 06-11 (morning peak)
    4.5, 4.0, 4.5, 5.0, 6.0, 7.0,   # 12-17 (afternoon)
    6.5, 5.5, 4.5, 3.5, 3.0, 2.5,   # 18-23 (evening decline)
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionPlan:
    """Execution plan for a single position."""
    symbol: str = ""
    position_usd: float = 0.0
    strategy: str = "TWAP"          # TWAP or VWAP
    n_slices: int = TWAP_DEFAULT_SLICES
    slice_sizes_usd: list[float] = field(default_factory=list)
    est_spread_cost: float = 0.0     # In USD
    est_slippage: float = 0.0        # In USD
    est_market_impact: float = 0.0   # In USD
    total_cost_usd: float = 0.0
    total_cost_bps: float = 0.0      # In basis points
    urgency: str = "normal"          # immediate / high / normal / low
    execution_window: str = "1h"     # immediate / 1h / 4h / 24h
    participation_rate: float = 0.0  # As fraction of daily volume
    adverse_move_prob: float = 0.0   # Probability of adverse price move


@dataclass
class ExecutionReport:
    """Full execution strategy report."""
    plans: list[ExecutionPlan] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_cost_bps: float = 0.0
    avg_urgency: str = "normal"
    recommended_strategy: str = "TWAP"
    total_position_usd: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SPREAD ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def _estimate_spread(signal: dict) -> float:
    """Estimate bid-ask spread in basis points from volatility and volume.

    Higher volatility + lower volume = wider spread.
    """
    vol = signal.get("volatility", 1.0) or 1.0
    volume = signal.get("volume_24h", 1_000_000) or 1_000_000
    asset_class = signal.get("asset_class", "crypto")

    # Base spread by asset class (in bps)
    base_spreads = {
        "crypto": 10.0,
        "forex": 2.0,
        "commodity": 5.0,
        "index": 3.0,
    }
    base = base_spreads.get(asset_class, 8.0)

    # Adjust for volatility (higher vol = wider spread)
    vol_adj = 1.0 + (vol / 100.0) * 2.0

    # Adjust for volume (lower volume = wider spread)
    vol_tier = 1.0
    if volume < 100_000:
        vol_tier = 3.0
    elif volume < 1_000_000:
        vol_tier = 2.0
    elif volume < 10_000_000:
        vol_tier = 1.5

    spread_bps = base * vol_adj * vol_tier
    return round(min(spread_bps, 100.0), 2)  # Cap at 1%


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET IMPACT MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def _market_impact(
    position_usd: float,
    daily_volume_usd: float,
    volatility_pct: float,
) -> tuple[float, float]:
    """Estimate market impact using square-root model.

    Impact = sigma * constant * sqrt(Q / V)

    Returns:
        (impact_bps, participation_rate)
    """
    if daily_volume_usd <= 0:
        return 50.0, 1.0

    participation = position_usd / daily_volume_usd
    vol = volatility_pct / 100.0

    impact_bps = vol * IMPACT_CONSTANT * np.sqrt(participation) * 10000
    impact_bps = round(min(impact_bps, 200.0), 2)  # Cap at 2%

    return impact_bps, round(participation, 6)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIPPAGE ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def _estimate_slippage(
    position_usd: float,
    daily_volume_usd: float,
    volatility_pct: float,
    n_slices: int,
) -> float:
    """Estimate execution slippage in basis points.

    Slippage increases with larger orders relative to volume.
    Splitting into more slices reduces per-slice slippage.
    """
    if daily_volume_usd <= 0:
        return 30.0

    slice_size = position_usd / max(n_slices, 1)
    slice_participation = slice_size / daily_volume_usd

    # Base slippage proportional to sqrt of participation
    base_slip = np.sqrt(slice_participation) * volatility_pct * 100

    return round(min(base_slip, 100.0), 2)


# ═══════════════════════════════════════════════════════════════════════════════
# URGENCY SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_urgency(signal: dict) -> tuple[str, str]:
    """Determine execution urgency and recommended window.

    Returns:
        (urgency_level, execution_window)
    """
    risk_score = signal.get("risk_score", 50) or 50
    direction = signal.get("direction", "neutral")
    vol = signal.get("volatility", 1.0) or 1.0

    score = 0

    # High risk → execute sooner
    if risk_score > 80:
        score += 3
    elif risk_score > 60:
        score += 2
    elif risk_score > 40:
        score += 1

    # Strong directional signal → execute sooner
    if direction in ("strong_up", "strong_down"):
        score += 2
    elif direction in ("up", "down"):
        score += 1

    # High volatility → execute sooner (before it gets worse)
    if vol > 5.0:
        score += 2
    elif vol > 3.0:
        score += 1

    if score >= 6:
        return "immediate", "immediate"
    elif score >= 4:
        return "high", "1h"
    elif score >= 2:
        return "normal", "4h"
    else:
        return "low", "24h"


# ═══════════════════════════════════════════════════════════════════════════════
# TWAP / VWAP SLICE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def _twap_slices(position_usd: float, n_slices: int) -> list[float]:
    """Generate equal TWAP slices."""
    slice_size = round(position_usd / n_slices, 2)
    slices = [slice_size] * n_slices
    # Adjust last slice for rounding
    slices[-1] = round(position_usd - slice_size * (n_slices - 1), 2)
    return slices


def _vwap_slices(position_usd: float, n_slices: int) -> list[float]:
    """Generate volume-weighted VWAP slices.

    Uses the hourly volume profile to weight slices.
    """
    # Map n_slices to volume profile
    profile = VWAP_VOLUME_PROFILE[:n_slices] if n_slices <= 24 else VWAP_VOLUME_PROFILE
    total_profile = sum(profile[:n_slices])

    if total_profile <= 0:
        return _twap_slices(position_usd, n_slices)

    slices = []
    remaining = position_usd
    for i in range(n_slices):
        weight = profile[i % len(profile)] / total_profile
        slice_amt = round(position_usd * weight, 2)
        slices.append(slice_amt)
        remaining -= slice_amt

    # Adjust last slice for rounding
    if slices:
        slices[-1] = round(slices[-1] + remaining, 2)

    return slices


# ═══════════════════════════════════════════════════════════════════════════════
# ADVERSE MOVE PROBABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def _adverse_move_probability(
    volatility_pct: float,
    execution_hours: float,
) -> float:
    """Estimate probability of adverse price move during execution window.

    Uses a simple normal distribution model.
    """
    hourly_vol = (volatility_pct / 100.0) / np.sqrt(24)
    window_vol = hourly_vol * np.sqrt(execution_hours)

    # Probability of > 0.5% adverse move
    threshold = 0.005
    if window_vol <= 0:
        return 0.0

    z = threshold / window_vol
    # Approximate Phi(-z) using logistic approximation
    prob = 1.0 / (1.0 + np.exp(1.7 * z))
    return round(min(max(prob, 0.0), 1.0), 3)


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD EXECUTION PLAN
# ═══════════════════════════════════════════════════════════════════════════════

def _build_plan(position: dict, signal: dict) -> ExecutionPlan:
    """Build execution plan for a single position."""
    symbol = position.get("symbol", signal.get("symbol", "UNKNOWN"))
    pos_usd = position.get("position_usd", 0.0)

    if pos_usd <= 0:
        return ExecutionPlan(symbol=symbol, position_usd=0.0)

    price = signal.get("price", 1.0) or 1.0
    vol = signal.get("volatility", 1.0) or 1.0
    volume = signal.get("volume_24h", 1_000_000) or 1_000_000
    daily_vol_usd = volume * price if volume < 1e12 else volume

    # Urgency
    urgency, window = _compute_urgency(signal)

    # Choose strategy
    if urgency == "immediate":
        strategy = "TWAP"
        n_slices = 4     # Fast execution, few slices
    elif urgency == "high":
        strategy = "VWAP"
        n_slices = 8
    elif urgency == "low":
        strategy = "TWAP"
        n_slices = 24    # Slow execution, many slices
    else:
        strategy = "VWAP"
        n_slices = TWAP_DEFAULT_SLICES

    # Generate slices
    if strategy == "TWAP":
        slices = _twap_slices(pos_usd, n_slices)
    else:
        slices = _vwap_slices(pos_usd, n_slices)

    # Cost estimation
    spread_bps = _estimate_spread(signal)
    spread_cost = pos_usd * spread_bps / 10000.0

    slippage_bps = _estimate_slippage(pos_usd, daily_vol_usd, vol, n_slices)
    slippage_cost = pos_usd * slippage_bps / 10000.0

    impact_bps, participation = _market_impact(pos_usd, daily_vol_usd, vol)
    impact_cost = pos_usd * impact_bps / 10000.0

    total_cost = spread_cost + slippage_cost + impact_cost
    total_bps = spread_bps + slippage_bps + impact_bps

    # Adverse move
    window_hours = {"immediate": 0.25, "1h": 1.0, "4h": 4.0, "24h": 24.0}.get(window, 1.0)
    adverse_prob = _adverse_move_probability(vol, window_hours)

    return ExecutionPlan(
        symbol=symbol,
        position_usd=round(pos_usd, 2),
        strategy=strategy,
        n_slices=n_slices,
        slice_sizes_usd=slices,
        est_spread_cost=round(spread_cost, 2),
        est_slippage=round(slippage_cost, 2),
        est_market_impact=round(impact_cost, 2),
        total_cost_usd=round(total_cost, 2),
        total_cost_bps=round(total_bps, 2),
        urgency=urgency,
        execution_window=window,
        participation_rate=participation,
        adverse_move_prob=adverse_prob,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT RUN
# ═══════════════════════════════════════════════════════════════════════════════

async def run(
    run_id: str,
    positions: list[dict],
    signals: list[dict],
) -> StepResult:
    """Execute the Execution Strategist: TWAP/VWAP planning.

    Args:
        run_id: Pipeline run identifier
        positions: Position dicts from position_sizer (symbol, position_usd, etc.)
        signals: Market signal dicts for volume/volatility data

    Returns:
        StepResult with ExecutionReport data
    """
    t0 = time.monotonic()
    console.print("[bold cyan]Agent Sentinel — Execution Strategist[/] planning order execution...")

    if not positions:
        latency = (time.monotonic() - t0) * 1000
        console.print("[red]Execution Strategist: no positions to execute[/]")
        return StepResult(
            step_name="execution_strategy",
            status=StepStatus.FAILED,
            error="No positions provided",
            agent_used="execution_strategist",
            latency_ms=latency,
        )

    # Build signal lookup
    signal_map = {}
    for s in signals:
        sym = s.get("symbol", "")
        ac = s.get("asset_class", "")
        signal_map[sym] = s
        signal_map[ac] = s

    # Build execution plans
    plans: list[ExecutionPlan] = []
    for pos in positions:
        symbol = pos.get("symbol", "")
        sig = signal_map.get(symbol, signals[0] if signals else {})
        plan = _build_plan(pos, sig)
        if plan.position_usd > 0:
            plans.append(plan)

    # Aggregate
    total_cost = sum(p.total_cost_usd for p in plans)
    total_position = sum(p.position_usd for p in plans)
    total_bps = (total_cost / max(total_position, 1)) * 10000 if total_position > 0 else 0

    urgency_scores = {"immediate": 4, "high": 3, "normal": 2, "low": 1}
    avg_urgency_score = np.mean([urgency_scores.get(p.urgency, 2) for p in plans]) if plans else 2
    if avg_urgency_score >= 3.5:
        avg_urgency = "immediate"
    elif avg_urgency_score >= 2.5:
        avg_urgency = "high"
    elif avg_urgency_score >= 1.5:
        avg_urgency = "normal"
    else:
        avg_urgency = "low"

    # Recommend overall strategy
    vwap_count = sum(1 for p in plans if p.strategy == "VWAP")
    recommended = "VWAP" if vwap_count > len(plans) / 2 else "TWAP"

    report = ExecutionReport(
        plans=plans,
        total_cost_usd=round(total_cost, 2),
        total_cost_bps=round(total_bps, 2),
        avg_urgency=avg_urgency,
        recommended_strategy=recommended,
        total_position_usd=round(total_position, 2),
    )

    latency = (time.monotonic() - t0) * 1000

    # === Audit ===
    db.save_audit(
        run_id, "execution_strategy", "execution_strategist",
        input_summary=f"{len(positions)} positions, {len(signals)} signals",
        output_summary=(
            f"{len(plans)} plans, total_cost={total_cost:.2f} USD ({total_bps:.1f} bps), "
            f"urgency={avg_urgency}, strategy={recommended}"
        ),
        model_used="twap_vwap",
        latency_ms=latency,
    )

    # === Display ===
    _print_report(report)

    confidence = 75 if len(plans) >= 3 else 60

    console.print(
        f"[green]Execution Strategist done[/] — {len(plans)} plans, "
        f"cost={total_bps:.1f} bps, urgency={avg_urgency}, {recommended} "
        f"in {int(latency)}ms"
    )

    return StepResult(
        step_name="execution_strategy",
        status=StepStatus.SUCCESS,
        data={
            "plans": [
                {
                    "symbol": p.symbol,
                    "position_usd": p.position_usd,
                    "strategy": p.strategy,
                    "n_slices": p.n_slices,
                    "est_spread_cost": p.est_spread_cost,
                    "est_slippage": p.est_slippage,
                    "est_market_impact": p.est_market_impact,
                    "total_cost_usd": p.total_cost_usd,
                    "total_cost_bps": p.total_cost_bps,
                    "urgency": p.urgency,
                    "execution_window": p.execution_window,
                    "participation_rate": p.participation_rate,
                    "adverse_move_prob": p.adverse_move_prob,
                }
                for p in plans
            ],
            "total_cost_usd": report.total_cost_usd,
            "total_cost_bps": report.total_cost_bps,
            "avg_urgency": report.avg_urgency,
            "recommended_strategy": report.recommended_strategy,
            "total_position_usd": report.total_position_usd,
            "plan_count": len(plans),
        },
        confidence=confidence,
        agent_used="execution_strategist",
        model_used="twap_vwap",
        latency_ms=latency,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def _print_report(report: ExecutionReport) -> None:
    """Display execution strategy results."""
    table = Table(title="Execution Strategy — TWAP/VWAP Planning", show_lines=False)
    table.add_column("Asset", style="cyan")
    table.add_column("Size USD", justify="right")
    table.add_column("Strategy", justify="center")
    table.add_column("Slices", justify="center")
    table.add_column("Spread", justify="right")
    table.add_column("Slippage", justify="right")
    table.add_column("Impact", justify="right")
    table.add_column("Total Cost", justify="right")
    table.add_column("Urgency", justify="center")
    table.add_column("Window", justify="center")

    for p in report.plans:
        urgency_color = {
            "immediate": "red",
            "high": "yellow",
            "normal": "green",
            "low": "dim",
        }.get(p.urgency, "white")

        cost_color = "green" if p.total_cost_bps < 20 else ("yellow" if p.total_cost_bps < 50 else "red")

        table.add_row(
            p.symbol,
            f"${p.position_usd:,.0f}",
            f"[bold]{p.strategy}[/]",
            str(p.n_slices),
            f"${p.est_spread_cost:.2f}",
            f"${p.est_slippage:.2f}",
            f"${p.est_market_impact:.2f}",
            f"[{cost_color}]${p.total_cost_usd:.2f}[/]",
            f"[{urgency_color}]{p.urgency.upper()}[/]",
            p.execution_window,
        )

    console.print(table)

    # Summary panel
    cost_color = "green" if report.total_cost_bps < 20 else ("yellow" if report.total_cost_bps < 50 else "red")
    console.print(Panel(
        f"[bold]Total Position:[/] ${report.total_position_usd:,.0f}\n"
        f"[bold]Total Execution Cost:[/] [{cost_color}]${report.total_cost_usd:.2f} ({report.total_cost_bps:.1f} bps)[/]\n"
        f"[bold]Average Urgency:[/] {report.avg_urgency.upper()}\n"
        f"[bold]Recommended Strategy:[/] {report.recommended_strategy}\n"
        f"[bold]Execution Plans:[/] {len(report.plans)}",
        title="[bold]Execution Strategist — Summary[/]",
        border_style="cyan",
    ))

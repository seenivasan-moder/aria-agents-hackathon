"""Agent Sentinel — Liquidity Analyzer: order book depth, bid-ask spreads, liquidity scoring.

Analyzes market liquidity conditions per asset and per asset class:
- Bid-ask spread estimation from volatility and volume proxies
- Order book depth scoring (0-100) based on volume_24h vs asset class medians
- Liquidity classification: "deep", "moderate", "thin", "illiquid"
- Market impact estimation: slippage for $10k, $50k, $100k notional orders
- Aggregate liquidity score per asset class
- Overall market liquidity index (volume-weighted average)

Enrichissement Airia best-effort via bridge (non-bloquant).
Utilise numpy pour les calculs statistiques.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.pipeline_engine import StepResult, StepStatus
from src.config import config
from src.airia_bridge import bridge
from src import database as db

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Liquidity analysis parameters
# ═══════════════════════════════════════════════════════════════════════════════

# Reference median volumes (USD 24h) by asset class for depth scoring
ASSET_CLASS_MEDIAN_VOLUME: dict[str, float] = {
    "crypto":    500_000_000.0,
    "forex":   2_000_000_000.0,
    "commodity": 200_000_000.0,
    "equity":    800_000_000.0,
}
DEFAULT_MEDIAN_VOLUME = 300_000_000.0

# Spread model coefficients — spread_bps = alpha * (volatility / sqrt(volume_ratio))
SPREAD_ALPHA = 12.0                 # Base coefficient (basis points)
SPREAD_MIN_BPS = 0.5                # Minimum spread (very liquid assets)
SPREAD_MAX_BPS = 500.0              # Maximum spread (illiquid assets)

# Depth score thresholds — depth = 100 * sigmoid(log(volume / median))
DEPTH_SIGMOID_STEEPNESS = 2.5       # Controls how fast the sigmoid transitions

# Liquidity classification thresholds (based on depth score)
LIQUIDITY_DEEP_THRESHOLD = 70.0
LIQUIDITY_MODERATE_THRESHOLD = 40.0
LIQUIDITY_THIN_THRESHOLD = 15.0

# Market impact model — Almgren-Chriss simplified
# slippage_pct = gamma * (order_size / adv)^delta * volatility
IMPACT_GAMMA = 0.15                 # Market impact coefficient
IMPACT_DELTA = 0.60                 # Power law exponent (sub-linear)
IMPACT_ORDER_SIZES = [10_000, 50_000, 100_000]  # USD notional test orders

# Minimum signals required for meaningful analysis
MIN_SIGNALS = 2


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LiquidityMetrics:
    """Liquidity metrics for a single asset."""
    symbol: str
    asset_class: str = "unknown"
    spread_bps: float = 0.0             # Estimated bid-ask spread in basis points
    depth_score: float = 0.0            # Order book depth score (0-100)
    classification: str = "moderate"     # deep / moderate / thin / illiquid
    volume_24h: float = 0.0
    volatility: float = 0.0
    impact_10k_pct: float = 0.0         # Slippage for $10k order
    impact_50k_pct: float = 0.0         # Slippage for $50k order
    impact_100k_pct: float = 0.0        # Slippage for $100k order
    liquidity_score: float = 0.0        # Composite score (0-100)


@dataclass
class LiquidityReport:
    """Complete liquidity analysis report."""
    asset_metrics: list[LiquidityMetrics] = field(default_factory=list)
    by_asset_class: dict[str, float] = field(default_factory=dict)
    overall_liquidity_index: float = 0.0
    deep_count: int = 0
    moderate_count: int = 0
    thin_count: int = 0
    illiquid_count: int = 0
    total_assets: int = 0
    airia_enriched: bool = False
    models_used: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# CALCULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _estimate_spread(volatility: float, volume_24h: float, asset_class: str) -> float:
    """Estimate bid-ask spread in basis points from volatility and volume.

    Uses a simplified market microstructure model:
        spread_bps = alpha * (volatility / sqrt(volume_ratio))

    Higher volatility widens spreads; higher volume narrows them.

    Args:
        volatility: Asset volatility (percentage, e.g. 2.5 for 2.5%)
        volume_24h: 24-hour trading volume in USD
        asset_class: Asset class for median volume reference

    Returns:
        Estimated bid-ask spread in basis points
    """
    median_vol = ASSET_CLASS_MEDIAN_VOLUME.get(asset_class, DEFAULT_MEDIAN_VOLUME)
    volume_ratio = max(volume_24h, 1.0) / median_vol

    # Avoid division by zero — very low volume_ratio means illiquid
    sqrt_ratio = np.sqrt(max(volume_ratio, 1e-6))

    # Use absolute volatility (handle negative edge cases)
    abs_vol = max(abs(volatility), 0.01)

    spread = SPREAD_ALPHA * (abs_vol / sqrt_ratio)
    spread = float(np.clip(spread, SPREAD_MIN_BPS, SPREAD_MAX_BPS))

    return round(spread, 2)


def _compute_depth_score(volume_24h: float, asset_class: str) -> float:
    """Compute order book depth score (0-100) based on volume relative to class median.

    Uses a sigmoid function on the log volume ratio for smooth scaling:
        depth = 100 * sigmoid(steepness * log(volume / median))

    Args:
        volume_24h: 24-hour trading volume in USD
        asset_class: Asset class for reference median

    Returns:
        Depth score 0-100
    """
    median_vol = ASSET_CLASS_MEDIAN_VOLUME.get(asset_class, DEFAULT_MEDIAN_VOLUME)
    ratio = max(volume_24h, 1.0) / median_vol

    # Sigmoid on log ratio: centers at ratio=1 (median), smooth transitions
    log_ratio = np.log(ratio)
    sigmoid_val = 1.0 / (1.0 + np.exp(-DEPTH_SIGMOID_STEEPNESS * log_ratio))

    depth = float(sigmoid_val * 100.0)
    return round(depth, 1)


def _classify_liquidity(depth_score: float) -> str:
    """Classify liquidity based on depth score.

    Returns:
        One of "deep", "moderate", "thin", "illiquid"
    """
    if depth_score >= LIQUIDITY_DEEP_THRESHOLD:
        return "deep"
    if depth_score >= LIQUIDITY_MODERATE_THRESHOLD:
        return "moderate"
    if depth_score >= LIQUIDITY_THIN_THRESHOLD:
        return "thin"
    return "illiquid"


def _estimate_market_impact(
    order_size_usd: float,
    volume_24h: float,
    volatility: float,
) -> float:
    """Estimate market impact (slippage) for a given order size.

    Simplified Almgren-Chriss model:
        slippage_pct = gamma * (order_size / adv)^delta * volatility

    Args:
        order_size_usd: Order notional in USD
        volume_24h: 24-hour average daily volume in USD
        volatility: Asset volatility (percentage)

    Returns:
        Estimated slippage in percentage
    """
    if volume_24h <= 0 or volatility <= 0:
        return round(order_size_usd / max(volume_24h, 1.0) * 100, 4)

    participation_rate = order_size_usd / max(volume_24h, 1.0)
    abs_vol = max(abs(volatility), 0.01) / 100.0  # Convert pct to decimal

    impact = IMPACT_GAMMA * (participation_rate ** IMPACT_DELTA) * abs_vol * 100.0
    impact = float(np.clip(impact, 0.0, 50.0))  # Cap at 50% slippage

    return round(impact, 4)


def _compute_composite_score(spread_bps: float, depth_score: float, impact_50k: float) -> float:
    """Compute a composite liquidity score (0-100) from multiple factors.

    Weighted combination:
        - 35% depth score (higher = better)
        - 35% inverse spread score (tighter = better)
        - 30% inverse impact score (lower impact = better)

    Args:
        spread_bps: Bid-ask spread in basis points
        depth_score: Depth score 0-100
        impact_50k: Slippage percentage for $50k order

    Returns:
        Composite liquidity score 0-100
    """
    # Spread score: 100 at 0.5 bps, 0 at 500 bps (log scale)
    spread_score = max(0.0, 100.0 - 20.0 * np.log1p(spread_bps))

    # Impact score: 100 at 0% slippage, decays exponentially
    impact_score = 100.0 * np.exp(-impact_50k * 10.0)

    composite = 0.35 * depth_score + 0.35 * spread_score + 0.30 * impact_score
    return round(float(np.clip(composite, 0.0, 100.0)), 1)


def _analyze_asset(signal: dict) -> LiquidityMetrics:
    """Compute all liquidity metrics for a single asset signal.

    Args:
        signal: Market signal dict with symbol, asset_class, volume_24h, volatility, price, etc.

    Returns:
        LiquidityMetrics for the asset
    """
    symbol = signal.get("symbol", "?")
    asset_class = signal.get("asset_class", "crypto")
    volume_24h = float(signal.get("volume_24h", 0) or 0)
    volatility = float(signal.get("volatility", 0) or 0)

    # Estimate spread
    spread = _estimate_spread(volatility, volume_24h, asset_class)

    # Depth score
    depth = _compute_depth_score(volume_24h, asset_class)

    # Classification
    classification = _classify_liquidity(depth)

    # Market impact for standard order sizes
    impact_10k = _estimate_market_impact(10_000, volume_24h, volatility)
    impact_50k = _estimate_market_impact(50_000, volume_24h, volatility)
    impact_100k = _estimate_market_impact(100_000, volume_24h, volatility)

    # Composite liquidity score
    liquidity_score = _compute_composite_score(spread, depth, impact_50k)

    return LiquidityMetrics(
        symbol=symbol,
        asset_class=asset_class,
        spread_bps=spread,
        depth_score=depth,
        classification=classification,
        volume_24h=volume_24h,
        volatility=volatility,
        impact_10k_pct=impact_10k,
        impact_50k_pct=impact_50k,
        impact_100k_pct=impact_100k,
        liquidity_score=liquidity_score,
    )


def _aggregate_by_asset_class(metrics: list[LiquidityMetrics]) -> dict[str, float]:
    """Compute average liquidity score per asset class.

    Args:
        metrics: List of per-asset liquidity metrics

    Returns:
        Dict mapping asset class name to average liquidity score
    """
    by_class: dict[str, list[float]] = {}
    for m in metrics:
        if m.asset_class not in by_class:
            by_class[m.asset_class] = []
        by_class[m.asset_class].append(m.liquidity_score)

    return {
        ac: round(float(np.mean(scores)), 1)
        for ac, scores in by_class.items()
    }


def _compute_overall_index(metrics: list[LiquidityMetrics]) -> float:
    """Compute overall market liquidity index as volume-weighted average.

    Assets with higher volume contribute more to the index.

    Args:
        metrics: List of per-asset liquidity metrics

    Returns:
        Overall liquidity index 0-100
    """
    if not metrics:
        return 0.0

    scores = np.array([m.liquidity_score for m in metrics], dtype=np.float64)
    volumes = np.array([max(m.volume_24h, 1.0) for m in metrics], dtype=np.float64)

    total_volume = np.sum(volumes)
    if total_volume <= 0:
        return round(float(np.mean(scores)), 1)

    weights = volumes / total_volume
    weighted_index = float(np.dot(scores, weights))

    return round(weighted_index, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# AIRIA ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _enrich_with_airia(
    asset_metrics: list[LiquidityMetrics],
    overall_index: float,
) -> dict[str, Any] | None:
    """Enrichissement via Airia — analyse qualitative de la liquidite (best-effort).

    Sends local liquidity analysis to Airia for supplementary narrative
    and market microstructure insights.

    Returns:
        Dict with Airia analysis or None if unavailable
    """
    if not bridge.is_available:
        return None

    try:
        input_data = {
            "liquidity_summary": {
                "overall_index": overall_index,
                "assets": [
                    {
                        "symbol": m.symbol,
                        "spread_bps": m.spread_bps,
                        "depth_score": m.depth_score,
                        "classification": m.classification,
                        "impact_50k_pct": m.impact_50k_pct,
                    }
                    for m in asset_metrics[:15]
                ],
            },
            "instruction": "Analyze this liquidity profile and provide market microstructure insights.",
        }

        result = bridge.execute_market_pipeline(
            [{"type": "liquidity_analysis", **input_data}]
        )

        if result.get("ok"):
            console.print(f"  [dim]Airia liquidity: {result.get('latency_ms', 0)}ms[/]")
            return result.get("parsed") or result.get("result")

        console.print(f"  [yellow]Airia liquidity failed: {result.get('error', '?')}[/]")
    except Exception as e:
        console.print(f"  [yellow]Airia liquidity error: {e}[/]")

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT RUN — Point d'entree principal
# ═══════════════════════════════════════════════════════════════════════════════

async def run(
    run_id: str,
    signals: list[dict],
) -> StepResult:
    """Execute l'agent Liquidity Analyzer: depth, spread, impact et scoring.

    Args:
        run_id: Identifiant du run pipeline
        signals: Liste de signaux marche (dicts avec symbol, asset_class,
                 volume_24h, volatility, price, change_24h, ...)

    Returns:
        StepResult avec LiquidityReport dans data
    """
    t0 = time.monotonic()
    console.print("[bold cyan]Agent Sentinel — Liquidity Analyzer[/] analyse en cours...")

    if not signals or len(signals) < MIN_SIGNALS:
        latency = (time.monotonic() - t0) * 1000
        console.print("[yellow]Liquidity Analyzer: pas assez de signaux[/]")
        return StepResult(
            step_name="liquidity_analysis",
            status=StepStatus.FAILED,
            error=f"Insuffisant: {len(signals or [])} signaux (min {MIN_SIGNALS})",
            agent_used="liquidity_analyzer",
            latency_ms=latency,
        )

    models_used = ["statistical"]

    # === Etape 1: Analyse par asset ===
    asset_metrics: list[LiquidityMetrics] = []
    for sig in signals:
        # Support both dict and dataclass-style objects
        if not isinstance(sig, dict):
            sig = sig.model_dump(mode="json") if hasattr(sig, "model_dump") else {"symbol": str(sig)}

        m = _analyze_asset(sig)
        asset_metrics.append(m)

    console.print(f"  [dim]{len(asset_metrics)} assets analyses[/]")

    # === Etape 2: Aggregation par classe ===
    by_class = _aggregate_by_asset_class(asset_metrics)
    for ac, score in sorted(by_class.items(), key=lambda x: x[1], reverse=True):
        console.print(f"    {ac}: liquidity={score:.0f}")

    # === Etape 3: Indice global ===
    overall_index = _compute_overall_index(asset_metrics)

    # === Etape 4: Classification counts ===
    deep_count = sum(1 for m in asset_metrics if m.classification == "deep")
    moderate_count = sum(1 for m in asset_metrics if m.classification == "moderate")
    thin_count = sum(1 for m in asset_metrics if m.classification == "thin")
    illiquid_count = sum(1 for m in asset_metrics if m.classification == "illiquid")

    # === Etape 5: Enrichissement Airia (best-effort) ===
    airia_result = _enrich_with_airia(asset_metrics, overall_index)
    airia_enriched = airia_result is not None
    if airia_enriched:
        models_used.append("airia")

    # === Build report ===
    report = LiquidityReport(
        asset_metrics=asset_metrics,
        by_asset_class=by_class,
        overall_liquidity_index=overall_index,
        deep_count=deep_count,
        moderate_count=moderate_count,
        thin_count=thin_count,
        illiquid_count=illiquid_count,
        total_assets=len(asset_metrics),
        airia_enriched=airia_enriched,
        models_used=models_used,
    )

    latency = (time.monotonic() - t0) * 1000

    # === Audit DB ===
    db.save_audit(
        run_id, "liquidity_analysis", "liquidity_analyzer",
        input_summary=f"{len(signals)} signaux, {len(by_class)} classes",
        output_summary=(
            f"index={overall_index:.0f}, deep={deep_count}, moderate={moderate_count}, "
            f"thin={thin_count}, illiquid={illiquid_count}, airia={'OUI' if airia_enriched else 'NON'}"
        ),
        model_used=", ".join(models_used),
        latency_ms=latency,
    )

    # === Affichage ===
    _print_liquidity_report(report)

    # Confidence: higher when market is liquid, lower when illiquid
    confidence = min(95.0, max(20.0, overall_index * 0.85 + 15.0))

    console.print(
        f"[green]Liquidity Analyzer done[/] — index={overall_index:.0f}, "
        f"{len(asset_metrics)} assets ({deep_count} deep, {illiquid_count} illiquid) "
        f"in {int(latency)}ms"
    )

    return StepResult(
        step_name="liquidity_analysis",
        status=StepStatus.SUCCESS,
        data={
            "assets": [
                {
                    "symbol": m.symbol,
                    "asset_class": m.asset_class,
                    "spread_bps": m.spread_bps,
                    "depth_score": m.depth_score,
                    "classification": m.classification,
                    "volume_24h": m.volume_24h,
                    "volatility": m.volatility,
                    "impact_10k_pct": m.impact_10k_pct,
                    "impact_50k_pct": m.impact_50k_pct,
                    "impact_100k_pct": m.impact_100k_pct,
                    "liquidity_score": m.liquidity_score,
                }
                for m in asset_metrics
            ],
            "by_asset_class": by_class,
            "overall_liquidity_index": overall_index,
            "classification_counts": {
                "deep": deep_count,
                "moderate": moderate_count,
                "thin": thin_count,
                "illiquid": illiquid_count,
            },
            "total_assets": report.total_assets,
            "airia_enriched": airia_enriched,
            "models_used": models_used,
        },
        confidence=round(confidence, 1),
        agent_used="liquidity_analyzer",
        model_used=", ".join(models_used),
        latency_ms=latency,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _classification_color(classification: str) -> str:
    """Return Rich color string for a liquidity classification."""
    return {
        "deep": "bold green",
        "moderate": "green",
        "thin": "yellow",
        "illiquid": "bold red",
    }.get(classification, "white")


def _spread_color(spread_bps: float) -> str:
    """Return Rich color for spread width."""
    if spread_bps < 5:
        return "green"
    if spread_bps < 20:
        return "yellow"
    if spread_bps < 100:
        return "red"
    return "bold red"


def _impact_color(impact_pct: float) -> str:
    """Return Rich color for market impact percentage."""
    if impact_pct < 0.01:
        return "green"
    if impact_pct < 0.05:
        return "yellow"
    if impact_pct < 0.5:
        return "red"
    return "bold red"


# ═══════════════════════════════════════════════════════════════════════════════
# AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _print_liquidity_report(report: LiquidityReport) -> None:
    """Affiche le rapport de liquidite complet."""
    if not report.asset_metrics:
        console.print("[yellow]Aucun asset analyse[/]")
        return

    # === Table principale: metriques par asset ===
    table = Table(
        title=f"Liquidity Analysis — {report.total_assets} Assets",
        show_lines=True,
    )
    table.add_column("Asset", style="cyan")
    table.add_column("Class", style="dim")
    table.add_column("Spread (bps)", justify="right")
    table.add_column("Depth", justify="right")
    table.add_column("Classification", justify="center")
    table.add_column("$10k Slip", justify="right")
    table.add_column("$50k Slip", justify="right")
    table.add_column("$100k Slip", justify="right")
    table.add_column("Score", justify="right")

    # Sort by liquidity score descending
    sorted_metrics = sorted(report.asset_metrics, key=lambda m: m.liquidity_score, reverse=True)

    for m in sorted_metrics[:20]:
        cls_color = _classification_color(m.classification)
        sp_color = _spread_color(m.spread_bps)
        depth_color = "green" if m.depth_score >= 60 else ("yellow" if m.depth_score >= 30 else "red")
        score_color = "green" if m.liquidity_score >= 60 else ("yellow" if m.liquidity_score >= 30 else "red")

        table.add_row(
            m.symbol,
            m.asset_class,
            f"[{sp_color}]{m.spread_bps:.1f}[/]",
            f"[{depth_color}]{m.depth_score:.0f}[/]",
            f"[{cls_color}]{m.classification.upper()}[/]",
            f"[{_impact_color(m.impact_10k_pct)}]{m.impact_10k_pct:.4f}%[/]",
            f"[{_impact_color(m.impact_50k_pct)}]{m.impact_50k_pct:.4f}%[/]",
            f"[{_impact_color(m.impact_100k_pct)}]{m.impact_100k_pct:.4f}%[/]",
            f"[{score_color}]{m.liquidity_score:.0f}[/]",
        )

    console.print(table)

    # === Table: aggregation par classe d'actifs ===
    if report.by_asset_class:
        class_table = Table(title="Liquidity by Asset Class", show_lines=False)
        class_table.add_column("Asset Class", style="cyan")
        class_table.add_column("Avg Liquidity Score", justify="right")
        class_table.add_column("Level")

        for ac, score in sorted(report.by_asset_class.items(), key=lambda x: x[1], reverse=True):
            color = "green" if score >= 60 else ("yellow" if score >= 30 else "red")
            level = "HIGH" if score >= 60 else ("MEDIUM" if score >= 30 else "LOW")
            class_table.add_row(ac, f"[{color}]{score:.0f}[/]", f"[{color}]{level}[/]")

        console.print(class_table)

    # === Panel de synthese ===
    idx = report.overall_liquidity_index
    if idx >= 60:
        border_color = "green"
        health_label = "HEALTHY"
    elif idx >= 30:
        border_color = "yellow"
        health_label = "MODERATE"
    else:
        border_color = "red"
        health_label = "STRESSED"

    console.print(Panel(
        f"[bold]Overall Liquidity Index:[/] [{border_color}]{idx:.0f}/100[/] "
        f"([{border_color}]{health_label}[/])\n"
        f"[bold]Assets:[/] {report.total_assets}\n"
        f"[bold green]Deep:[/] {report.deep_count}  "
        f"[bold]Moderate:[/] {report.moderate_count}  "
        f"[bold yellow]Thin:[/] {report.thin_count}  "
        f"[bold red]Illiquid:[/] {report.illiquid_count}\n"
        f"[bold]Airia Enriched:[/] {'OUI' if report.airia_enriched else 'NON'}\n"
        f"[bold]Modeles:[/] {', '.join(report.models_used)}",
        title="[bold]Liquidity Analyzer — Synthese[/]",
        border_style=border_color,
    ))

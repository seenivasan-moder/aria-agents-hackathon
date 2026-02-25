"""Agent Sentinel — Volatility Forecaster: EWMA volatility and Value-at-Risk estimation.

Forecasts forward-looking volatility and computes risk metrics per asset:
- EWMA (Exponentially Weighted Moving Average) volatility with lambda=0.94
- Historical volatility estimation from change_1h, change_24h, change_7d
- Next-period volatility forecast per asset
- Parametric VaR at 95% and 99% confidence levels
- CVaR / Expected Shortfall (Conditional VaR)
- Volatility regime classification: "low", "normal", "high", "extreme"
- Aggregate portfolio VaR (equal weights, correlation-adjusted)

Enrichissement Airia best-effort via bridge (non-bloquant).
Utilise numpy pour les calculs statistiques et matriciels.
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
# CONFIGURATION — EWMA & VaR parameters
# ═══════════════════════════════════════════════════════════════════════════════

# EWMA decay factor (RiskMetrics standard)
EWMA_LAMBDA = 0.94

# VaR confidence z-scores (standard normal quantiles)
VAR_95_ZSCORE = 1.6449              # scipy.stats.norm.ppf(0.95)
VAR_99_ZSCORE = 2.3263              # scipy.stats.norm.ppf(0.99)

# CVaR multipliers: E[X | X > VaR_z] = sigma * phi(z) / (1-alpha)
# where phi is the standard normal PDF
CVAR_95_MULTIPLIER = 2.0627         # phi(1.6449) / 0.05
CVAR_99_MULTIPLIER = 2.6652         # phi(2.3263) / 0.01

# Volatility regime thresholds (daily volatility in percent)
VOL_LOW_MAX = 1.0                   # < 1%  daily vol = low
VOL_NORMAL_MAX = 3.0                # 1-3%  daily vol = normal
VOL_HIGH_MAX = 5.0                  # 3-5%  daily vol = high
# > 5% = extreme

# Portfolio VaR
DEFAULT_PORTFOLIO_VALUE = 1_000_000.0   # $1M reference portfolio
EQUAL_WEIGHT_FALLBACK = True             # Equal weights if no allocation provided

# Timeframe conversion factors (to daily volatility)
# Assume: change_1h ~ hourly return -> annualize by sqrt(24*365) then daily by /sqrt(365)
# change_24h ~ daily return (direct)
# change_7d ~ weekly return -> daily by /sqrt(7)
TIMEFRAME_TO_DAILY = {
    "change_1h":  np.sqrt(24.0),    # Hourly to daily: * sqrt(24)
    "change_24h": 1.0,               # Already daily
    "change_7d":  1.0 / np.sqrt(7),  # Weekly to daily: / sqrt(7)
}

# Timeframe weights for composite volatility estimate
TIMEFRAME_VOL_WEIGHTS = {
    "change_1h":  0.30,
    "change_24h": 0.50,
    "change_7d":  0.20,
}

# Minimum number of signals
MIN_SIGNALS = 2


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VolatilityForecast:
    """Volatility forecast and VaR for a single asset."""
    symbol: str
    asset_class: str = "unknown"
    historical_vol_pct: float = 0.0     # Historical daily volatility (%)
    ewma_vol_pct: float = 0.0           # EWMA forecasted daily volatility (%)
    forecast_vol_pct: float = 0.0       # Final forecasted vol (blend of hist + EWMA)
    var_95_pct: float = 0.0             # VaR 95% in % of position value
    var_99_pct: float = 0.0             # VaR 99% in %
    cvar_95_pct: float = 0.0            # CVaR 95% (Expected Shortfall) in %
    cvar_99_pct: float = 0.0            # CVaR 99% in %
    regime: str = "normal"              # low / normal / high / extreme
    annualized_vol_pct: float = 0.0     # Annualized volatility (%)
    price: float = 0.0
    change_24h: float = 0.0


@dataclass
class PortfolioVaR:
    """Portfolio-level VaR summary."""
    portfolio_value: float = DEFAULT_PORTFOLIO_VALUE
    num_assets: int = 0
    portfolio_var_95_pct: float = 0.0
    portfolio_var_95_usd: float = 0.0
    portfolio_var_99_pct: float = 0.0
    portfolio_var_99_usd: float = 0.0
    portfolio_cvar_95_pct: float = 0.0
    portfolio_cvar_95_usd: float = 0.0
    avg_correlation: float = 0.0
    diversification_ratio: float = 0.0  # Undiversified / Diversified VaR
    method: str = "parametric-ewma"


@dataclass
class VaRReport:
    """Complete volatility and VaR report."""
    forecasts: list[VolatilityForecast] = field(default_factory=list)
    portfolio_var: PortfolioVaR = field(default_factory=PortfolioVaR)
    regime_distribution: dict[str, int] = field(default_factory=dict)
    avg_vol_pct: float = 0.0
    max_vol_asset: str = ""
    max_vol_pct: float = 0.0
    total_assets: int = 0
    airia_enriched: bool = False
    models_used: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# VOLATILITY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _estimate_historical_vol(signal: dict) -> float:
    """Estimate historical daily volatility from multi-timeframe returns.

    Combines change_1h, change_24h, change_7d with appropriate scaling
    factors and weights to produce a single daily volatility estimate.

    Args:
        signal: Market signal dict with change_1h, change_24h, change_7d

    Returns:
        Estimated daily volatility in percentage
    """
    weighted_sum = 0.0
    total_weight = 0.0

    for tf, weight in TIMEFRAME_VOL_WEIGHTS.items():
        raw_change = float(signal.get(tf, 0) or 0)
        daily_factor = TIMEFRAME_TO_DAILY[tf]

        # Absolute return scaled to daily frequency
        daily_vol_contribution = abs(raw_change) * daily_factor
        weighted_sum += daily_vol_contribution * weight
        total_weight += weight

    if total_weight <= 0:
        return 0.0

    return weighted_sum / total_weight


def _compute_ewma_vol(
    historical_vol: float,
    previous_ewma: float | None = None,
    decay: float = EWMA_LAMBDA,
) -> float:
    """Compute EWMA volatility using the RiskMetrics approach.

    Formula: sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * r_{t-1}^2

    Since we have a single snapshot, we use the historical vol as both
    the previous EWMA and the latest return shock.

    Args:
        historical_vol: Current period historical volatility (%)
        previous_ewma: Previous period EWMA vol (%, or None for initialization)
        decay: EWMA decay factor (0.94 default)

    Returns:
        EWMA volatility forecast in percentage
    """
    if previous_ewma is None:
        # Initialize EWMA with historical vol (no prior available)
        previous_ewma = historical_vol

    # EWMA variance
    ewma_var = decay * (previous_ewma ** 2) + (1 - decay) * (historical_vol ** 2)
    ewma_vol = np.sqrt(max(ewma_var, 0.0))

    return float(ewma_vol)


def _blend_volatility(historical: float, ewma: float, alpha: float = 0.6) -> float:
    """Blend historical and EWMA volatility for final forecast.

    Args:
        historical: Historical volatility (%)
        ewma: EWMA volatility (%)
        alpha: Weight on EWMA (higher = more weight on EWMA)

    Returns:
        Blended forecast volatility (%)
    """
    return alpha * ewma + (1 - alpha) * historical


def _compute_var(vol_pct: float, z_score: float) -> float:
    """Compute parametric VaR as percentage of position value.

    VaR = z * sigma (assuming zero mean for short horizons)

    Args:
        vol_pct: Volatility in percentage
        z_score: Z-score for confidence level

    Returns:
        VaR in percentage (positive = potential loss)
    """
    return abs(vol_pct * z_score)


def _compute_cvar(vol_pct: float, cvar_multiplier: float) -> float:
    """Compute Conditional VaR (Expected Shortfall) as percentage.

    CVaR = sigma * phi(z) / (1 - alpha) where phi is the standard normal PDF.

    Args:
        vol_pct: Volatility in percentage
        cvar_multiplier: Pre-computed phi(z) / (1-alpha) for the confidence level

    Returns:
        CVaR in percentage
    """
    return abs(vol_pct * cvar_multiplier)


def _classify_vol_regime(daily_vol_pct: float) -> str:
    """Classify volatility regime based on daily volatility.

    Returns:
        One of "low", "normal", "high", "extreme"
    """
    if daily_vol_pct < VOL_LOW_MAX:
        return "low"
    if daily_vol_pct < VOL_NORMAL_MAX:
        return "normal"
    if daily_vol_pct < VOL_HIGH_MAX:
        return "high"
    return "extreme"


def _forecast_asset(signal: dict) -> VolatilityForecast:
    """Compute complete volatility forecast and VaR for a single asset.

    Args:
        signal: Market signal dict

    Returns:
        VolatilityForecast with all metrics
    """
    symbol = signal.get("symbol", "?")
    asset_class = signal.get("asset_class", "crypto")
    price = float(signal.get("price", 0) or 0)
    change_24h = float(signal.get("change_24h", 0) or 0)

    # Historical volatility
    hist_vol = _estimate_historical_vol(signal)

    # EWMA forecast
    ewma_vol = _compute_ewma_vol(hist_vol)

    # Blended forecast
    forecast_vol = _blend_volatility(hist_vol, ewma_vol)

    # VaR
    var_95 = _compute_var(forecast_vol, VAR_95_ZSCORE)
    var_99 = _compute_var(forecast_vol, VAR_99_ZSCORE)

    # CVaR
    cvar_95 = _compute_cvar(forecast_vol, CVAR_95_MULTIPLIER)
    cvar_99 = _compute_cvar(forecast_vol, CVAR_99_MULTIPLIER)

    # Regime
    regime = _classify_vol_regime(forecast_vol)

    # Annualized
    annualized = forecast_vol * np.sqrt(365)

    return VolatilityForecast(
        symbol=symbol,
        asset_class=asset_class,
        historical_vol_pct=round(hist_vol, 4),
        ewma_vol_pct=round(ewma_vol, 4),
        forecast_vol_pct=round(forecast_vol, 4),
        var_95_pct=round(var_95, 4),
        var_99_pct=round(var_99, 4),
        cvar_95_pct=round(cvar_95, 4),
        cvar_99_pct=round(cvar_99, 4),
        regime=regime,
        annualized_vol_pct=round(float(annualized), 2),
        price=price,
        change_24h=change_24h,
    )


def _compute_portfolio_var(
    forecasts: list[VolatilityForecast],
    correlation_matrix: np.ndarray | None = None,
    portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
) -> PortfolioVaR:
    """Compute portfolio-level VaR assuming equal weights.

    Uses the variance-covariance approach:
        portfolio_var = sqrt(w' * Sigma * w) * z

    If no correlation matrix is provided, estimates one from the
    signal data (simplified approach).

    Args:
        forecasts: List of per-asset volatility forecasts
        correlation_matrix: Optional pre-computed correlation matrix
        portfolio_value: Total portfolio value in USD

    Returns:
        PortfolioVaR with diversified metrics
    """
    n = len(forecasts)
    if n == 0:
        return PortfolioVaR(portfolio_value=portfolio_value)

    # Equal weights
    weights = np.ones(n) / n

    # Volatility vector (daily, as decimals)
    vols = np.array([f.forecast_vol_pct / 100.0 for f in forecasts], dtype=np.float64)

    # Build or use correlation matrix
    if correlation_matrix is not None and correlation_matrix.shape == (n, n):
        corr = correlation_matrix
    else:
        # Estimate simple correlation from co-movement of changes
        changes = np.array([f.change_24h for f in forecasts], dtype=np.float64)
        if n >= 2 and np.std(changes) > 1e-10:
            # Sign-based correlation proxy
            signs = np.sign(changes)
            corr = np.outer(signs, signs) * 0.3 + np.eye(n) * 0.7
            np.fill_diagonal(corr, 1.0)
        else:
            corr = np.eye(n)

    # Covariance matrix: Sigma = diag(vol) * Corr * diag(vol)
    vol_diag = np.diag(vols)
    cov_matrix = vol_diag @ corr @ vol_diag

    # Portfolio variance
    port_var = float(weights @ cov_matrix @ weights)
    port_vol = np.sqrt(max(port_var, 0.0))

    # Portfolio VaR
    port_var_95 = float(port_vol * VAR_95_ZSCORE * 100)  # Back to percentage
    port_var_99 = float(port_vol * VAR_99_ZSCORE * 100)
    port_cvar_95 = float(port_vol * CVAR_95_MULTIPLIER * 100)

    # USD amounts
    var_95_usd = port_var_95 / 100 * portfolio_value
    var_99_usd = port_var_99 / 100 * portfolio_value
    cvar_95_usd = port_cvar_95 / 100 * portfolio_value

    # Average correlation (upper triangle)
    upper = corr[np.triu_indices(n, k=1)]
    avg_corr = float(np.mean(upper)) if len(upper) > 0 else 0.0

    # Diversification ratio: undiversified VaR / diversified VaR
    undiversified_var = float(np.sum(weights * vols) * VAR_95_ZSCORE * 100)
    div_ratio = undiversified_var / max(port_var_95, 1e-10) if port_var_95 > 0 else 1.0

    return PortfolioVaR(
        portfolio_value=portfolio_value,
        num_assets=n,
        portfolio_var_95_pct=round(port_var_95, 4),
        portfolio_var_95_usd=round(var_95_usd, 2),
        portfolio_var_99_pct=round(port_var_99, 4),
        portfolio_var_99_usd=round(var_99_usd, 2),
        portfolio_cvar_95_pct=round(port_cvar_95, 4),
        portfolio_cvar_95_usd=round(cvar_95_usd, 2),
        avg_correlation=round(avg_corr, 4),
        diversification_ratio=round(float(div_ratio), 3),
        method="parametric-ewma",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AIRIA ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _enrich_with_airia(
    forecasts: list[VolatilityForecast],
    portfolio_var: PortfolioVaR,
) -> dict[str, Any] | None:
    """Enrichissement via Airia — analyse narrative volatilite (best-effort).

    Sends local VaR analysis to Airia for supplementary narrative
    insights and regime interpretation.

    Returns:
        Dict with Airia analysis or None if unavailable
    """
    if not bridge.is_available:
        return None

    try:
        input_data = {
            "volatility_summary": {
                "portfolio_var_95_pct": portfolio_var.portfolio_var_95_pct,
                "portfolio_var_99_pct": portfolio_var.portfolio_var_99_pct,
                "avg_correlation": portfolio_var.avg_correlation,
                "assets": [
                    {
                        "symbol": f.symbol,
                        "forecast_vol_pct": f.forecast_vol_pct,
                        "var_95_pct": f.var_95_pct,
                        "regime": f.regime,
                    }
                    for f in forecasts[:15]
                ],
            },
            "instruction": "Analyze this volatility profile and assess portfolio risk.",
        }

        result = bridge.execute_risk_aggregation(input_data)

        if result.get("ok"):
            console.print(f"  [dim]Airia volatility: {result.get('latency_ms', 0)}ms[/]")
            return result.get("parsed") or result.get("result")

        console.print(f"  [yellow]Airia volatility failed: {result.get('error', '?')}[/]")
    except Exception as e:
        console.print(f"  [yellow]Airia volatility error: {e}[/]")

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT RUN — Point d'entree principal
# ═══════════════════════════════════════════════════════════════════════════════

async def run(
    run_id: str,
    signals: list[dict],
) -> StepResult:
    """Execute l'agent Volatility Forecaster: EWMA vol + VaR parametrique.

    Args:
        run_id: Identifiant du run pipeline
        signals: Liste de signaux marche (dicts avec symbol, asset_class,
                 change_1h, change_24h, change_7d, volatility, price, ...)

    Returns:
        StepResult avec VaRReport dans data
    """
    t0 = time.monotonic()
    console.print("[bold cyan]Agent Sentinel — Volatility Forecaster[/] analyse en cours...")

    if not signals or len(signals) < MIN_SIGNALS:
        latency = (time.monotonic() - t0) * 1000
        console.print("[yellow]Volatility Forecaster: pas assez de signaux[/]")
        return StepResult(
            step_name="volatility_forecast",
            status=StepStatus.FAILED,
            error=f"Insuffisant: {len(signals or [])} signaux (min {MIN_SIGNALS})",
            agent_used="volatility_forecaster",
            latency_ms=latency,
        )

    models_used = ["ewma_var"]

    # Normalize input
    signal_dicts: list[dict] = []
    for s in signals:
        if isinstance(s, dict):
            signal_dicts.append(s)
        elif hasattr(s, "model_dump"):
            signal_dicts.append(s.model_dump(mode="json"))
        else:
            signal_dicts.append({"symbol": str(s)})

    # === Etape 1: Per-asset volatility forecast ===
    forecasts: list[VolatilityForecast] = []
    for sig in signal_dicts:
        f = _forecast_asset(sig)
        forecasts.append(f)

    console.print(f"  [dim]{len(forecasts)} assets forecasted[/]")

    # === Etape 2: Regime distribution ===
    regime_counts: dict[str, int] = {"low": 0, "normal": 0, "high": 0, "extreme": 0}
    for f in forecasts:
        regime_counts[f.regime] = regime_counts.get(f.regime, 0) + 1

    for regime, count in regime_counts.items():
        if count > 0:
            console.print(f"    {regime}: {count} assets")

    # === Etape 3: Average and max volatility ===
    vols = [f.forecast_vol_pct for f in forecasts]
    avg_vol = float(np.mean(vols)) if vols else 0.0
    max_vol_idx = int(np.argmax(vols)) if vols else 0
    max_vol_asset = forecasts[max_vol_idx].symbol if forecasts else ""
    max_vol = forecasts[max_vol_idx].forecast_vol_pct if forecasts else 0.0

    console.print(f"  [dim]Avg vol: {avg_vol:.2f}%, max: {max_vol_asset} ({max_vol:.2f}%)[/]")

    # === Etape 4: Portfolio VaR ===
    portfolio_var = _compute_portfolio_var(forecasts)
    console.print(
        f"  [dim]Portfolio VaR 95%: {portfolio_var.portfolio_var_95_pct:.3f}% "
        f"(${portfolio_var.portfolio_var_95_usd:,.0f}), "
        f"div_ratio={portfolio_var.diversification_ratio:.2f}[/]"
    )

    # === Etape 5: Airia enrichment (best-effort) ===
    airia_result = _enrich_with_airia(forecasts, portfolio_var)
    airia_enriched = airia_result is not None
    if airia_enriched:
        models_used.append("airia")

    # === Build report ===
    report = VaRReport(
        forecasts=forecasts,
        portfolio_var=portfolio_var,
        regime_distribution=regime_counts,
        avg_vol_pct=round(avg_vol, 4),
        max_vol_asset=max_vol_asset,
        max_vol_pct=round(max_vol, 4),
        total_assets=len(forecasts),
        airia_enriched=airia_enriched,
        models_used=models_used,
    )

    latency = (time.monotonic() - t0) * 1000

    # === Audit DB ===
    db.save_audit(
        run_id, "volatility_forecast", "volatility_forecaster",
        input_summary=f"{len(signals)} signaux, EWMA lambda={EWMA_LAMBDA}",
        output_summary=(
            f"avg_vol={avg_vol:.2f}%, max={max_vol_asset}({max_vol:.2f}%), "
            f"VaR95={portfolio_var.portfolio_var_95_pct:.3f}% "
            f"(${portfolio_var.portfolio_var_95_usd:,.0f}), "
            f"regimes={regime_counts}, airia={'OUI' if airia_enriched else 'NON'}"
        ),
        model_used=", ".join(models_used),
        latency_ms=latency,
    )

    # === Affichage ===
    _print_var_report(report)

    # Confidence: higher when volatility is low and predictable
    extreme_count = regime_counts.get("extreme", 0)
    high_count = regime_counts.get("high", 0)
    if extreme_count > 0:
        confidence = max(30.0, 60.0 - extreme_count * 5.0)
    elif high_count > 0:
        confidence = max(50.0, 75.0 - high_count * 3.0)
    else:
        confidence = min(92.0, 85.0 + (1.0 / max(avg_vol, 0.1)))

    console.print(
        f"[green]Volatility Forecaster done[/] — avg_vol={avg_vol:.2f}%, "
        f"VaR95={portfolio_var.portfolio_var_95_pct:.3f}% "
        f"(${portfolio_var.portfolio_var_95_usd:,.0f}), "
        f"{len(forecasts)} assets in {int(latency)}ms"
    )

    return StepResult(
        step_name="volatility_forecast",
        status=StepStatus.SUCCESS,
        data={
            "forecasts": [
                {
                    "symbol": f.symbol,
                    "asset_class": f.asset_class,
                    "historical_vol_pct": f.historical_vol_pct,
                    "ewma_vol_pct": f.ewma_vol_pct,
                    "forecast_vol_pct": f.forecast_vol_pct,
                    "var_95_pct": f.var_95_pct,
                    "var_99_pct": f.var_99_pct,
                    "cvar_95_pct": f.cvar_95_pct,
                    "cvar_99_pct": f.cvar_99_pct,
                    "regime": f.regime,
                    "annualized_vol_pct": f.annualized_vol_pct,
                    "price": f.price,
                    "change_24h": f.change_24h,
                }
                for f in forecasts
            ],
            "portfolio_var": {
                "portfolio_value": portfolio_var.portfolio_value,
                "num_assets": portfolio_var.num_assets,
                "var_95_pct": portfolio_var.portfolio_var_95_pct,
                "var_95_usd": portfolio_var.portfolio_var_95_usd,
                "var_99_pct": portfolio_var.portfolio_var_99_pct,
                "var_99_usd": portfolio_var.portfolio_var_99_usd,
                "cvar_95_pct": portfolio_var.portfolio_cvar_95_pct,
                "cvar_95_usd": portfolio_var.portfolio_cvar_95_usd,
                "avg_correlation": portfolio_var.avg_correlation,
                "diversification_ratio": portfolio_var.diversification_ratio,
                "method": portfolio_var.method,
            },
            "regime_distribution": regime_counts,
            "avg_vol_pct": report.avg_vol_pct,
            "max_vol_asset": report.max_vol_asset,
            "max_vol_pct": report.max_vol_pct,
            "total_assets": report.total_assets,
            "airia_enriched": airia_enriched,
            "models_used": models_used,
        },
        confidence=round(confidence, 1),
        agent_used="volatility_forecaster",
        model_used=", ".join(models_used),
        latency_ms=latency,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _regime_color(regime: str) -> str:
    """Return Rich color for a volatility regime."""
    return {
        "low": "green",
        "normal": "cyan",
        "high": "yellow",
        "extreme": "bold red",
    }.get(regime, "white")


def _var_color(var_pct: float) -> str:
    """Return Rich color for a VaR percentage."""
    if var_pct < 2.0:
        return "green"
    if var_pct < 5.0:
        return "yellow"
    if var_pct < 10.0:
        return "red"
    return "bold red"


def _vol_color(vol_pct: float) -> str:
    """Return Rich color for a volatility percentage."""
    if vol_pct < VOL_LOW_MAX:
        return "green"
    if vol_pct < VOL_NORMAL_MAX:
        return "cyan"
    if vol_pct < VOL_HIGH_MAX:
        return "yellow"
    return "bold red"


# ═══════════════════════════════════════════════════════════════════════════════
# AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _print_var_report(report: VaRReport) -> None:
    """Affiche le rapport VaR complet."""
    if not report.forecasts:
        console.print("[yellow]Aucune prevision de volatilite[/]")
        return

    # === Table: Per-asset volatility and VaR ===
    table = Table(
        title=f"Volatility Forecast & VaR — {report.total_assets} Assets (EWMA lambda={EWMA_LAMBDA})",
        show_lines=True,
    )
    table.add_column("Asset", style="cyan")
    table.add_column("Class", style="dim")
    table.add_column("Hist Vol", justify="right")
    table.add_column("EWMA Vol", justify="right")
    table.add_column("Forecast", justify="right")
    table.add_column("VaR 95%", justify="right")
    table.add_column("VaR 99%", justify="right")
    table.add_column("CVaR 95%", justify="right")
    table.add_column("Regime", justify="center")

    # Sort by forecast vol descending (most volatile first)
    sorted_forecasts = sorted(report.forecasts, key=lambda f: f.forecast_vol_pct, reverse=True)

    for f in sorted_forecasts[:20]:
        r_color = _regime_color(f.regime)
        v_color = _vol_color(f.forecast_vol_pct)
        var95_color = _var_color(f.var_95_pct)
        var99_color = _var_color(f.var_99_pct)

        table.add_row(
            f.symbol,
            f.asset_class,
            f"{f.historical_vol_pct:.2f}%",
            f"{f.ewma_vol_pct:.2f}%",
            f"[{v_color}]{f.forecast_vol_pct:.2f}%[/]",
            f"[{var95_color}]{f.var_95_pct:.2f}%[/]",
            f"[{var99_color}]{f.var_99_pct:.2f}%[/]",
            f"{f.cvar_95_pct:.2f}%",
            f"[{r_color}]{f.regime.upper()}[/]",
        )

    console.print(table)

    # === Table: Regime distribution ===
    regime_table = Table(title="Volatility Regime Distribution", show_lines=False)
    regime_table.add_column("Regime", style="cyan")
    regime_table.add_column("Count", justify="right")
    regime_table.add_column("% of Total", justify="right")

    total = max(report.total_assets, 1)
    for regime in ["low", "normal", "high", "extreme"]:
        count = report.regime_distribution.get(regime, 0)
        if count > 0:
            color = _regime_color(regime)
            pct = count / total * 100
            regime_table.add_row(
                f"[{color}]{regime.upper()}[/]",
                f"[{color}]{count}[/]",
                f"{pct:.0f}%",
            )

    console.print(regime_table)

    # === Panel: Portfolio VaR Summary ===
    pv = report.portfolio_var
    var_border = "green"
    if pv.portfolio_var_95_pct >= 5.0:
        var_border = "red"
    elif pv.portfolio_var_95_pct >= 2.0:
        var_border = "yellow"

    console.print(Panel(
        f"[bold]Portfolio Value:[/] ${pv.portfolio_value:,.0f}\n"
        f"[bold]Assets:[/] {pv.num_assets}\n"
        f"\n"
        f"[bold]VaR 95%:[/] [{_var_color(pv.portfolio_var_95_pct)}]"
        f"{pv.portfolio_var_95_pct:.3f}%[/] "
        f"(${pv.portfolio_var_95_usd:,.0f})\n"
        f"[bold]VaR 99%:[/] [{_var_color(pv.portfolio_var_99_pct)}]"
        f"{pv.portfolio_var_99_pct:.3f}%[/] "
        f"(${pv.portfolio_var_99_usd:,.0f})\n"
        f"[bold]CVaR 95%:[/] {pv.portfolio_cvar_95_pct:.3f}% "
        f"(${pv.portfolio_cvar_95_usd:,.0f})\n"
        f"\n"
        f"[bold]Avg Correlation:[/] {pv.avg_correlation:.3f}\n"
        f"[bold]Diversification Ratio:[/] {pv.diversification_ratio:.2f}x\n"
        f"[bold]Method:[/] {pv.method}\n"
        f"\n"
        f"[bold]Avg Volatility:[/] [{_vol_color(report.avg_vol_pct)}]"
        f"{report.avg_vol_pct:.2f}%[/]\n"
        f"[bold]Most Volatile:[/] {report.max_vol_asset} "
        f"([{_vol_color(report.max_vol_pct)}]{report.max_vol_pct:.2f}%[/])\n"
        f"[bold]Airia Enriched:[/] {'OUI' if report.airia_enriched else 'NON'}\n"
        f"[bold]Modeles:[/] {', '.join(report.models_used)}",
        title="[bold]Volatility Forecaster — Portfolio VaR Summary[/]",
        border_style=var_border,
    ))

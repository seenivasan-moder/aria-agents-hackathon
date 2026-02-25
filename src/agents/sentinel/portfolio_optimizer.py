"""Agent Sentinel — Portfolio Optimizer: Markowitz mean-variance optimization.

Implements:
- Expected returns estimation from market signals
- Covariance matrix construction from volatility/correlation
- Minimum variance portfolio
- Maximum Sharpe ratio portfolio (risk-free rate 4% annual)
- Efficient frontier (20 points)
- Comparison with current allocation

Uses numpy for matrix algebra.
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

RISK_FREE_RATE = 0.04  # 4% annual
MIN_WEIGHT = 0.01       # 1% minimum per asset
MAX_WEIGHT = 0.25       # 25% maximum per asset
FRONTIER_POINTS = 20    # Number of efficient frontier points
TRADING_DAYS = 252      # Annualization factor


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptimalPortfolio:
    """Result of a single optimization."""
    name: str = ""
    weights: dict[str, float] = field(default_factory=dict)
    expected_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0


@dataclass
class EfficientFrontierPoint:
    """A point on the efficient frontier."""
    target_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizerReport:
    """Full optimization report."""
    min_variance: OptimalPortfolio = field(default_factory=OptimalPortfolio)
    max_sharpe: OptimalPortfolio = field(default_factory=OptimalPortfolio)
    equal_weight: OptimalPortfolio = field(default_factory=OptimalPortfolio)
    frontier: list[EfficientFrontierPoint] = field(default_factory=list)
    asset_names: list[str] = field(default_factory=list)
    expected_returns: dict[str, float] = field(default_factory=dict)
    tracking_error: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# EXPECTED RETURNS
# ═══════════════════════════════════════════════════════════════════════════════

def _estimate_returns(signals: list[dict]) -> tuple[list[str], np.ndarray]:
    """Estimate expected returns from signal data.

    Uses change_24h as a short-term return proxy, annualized.
    """
    names = []
    returns = []

    for s in signals:
        symbol = s.get("symbol", "UNKNOWN")
        change_24h = s.get("change_24h", 0.0) or 0.0
        # Annualize daily return
        daily_return = change_24h / 100.0
        annual_return = daily_return * TRADING_DAYS
        # Clamp to reasonable range
        annual_return = max(-0.5, min(2.0, annual_return))
        names.append(symbol)
        returns.append(annual_return)

    return names, np.array(returns)


# ═══════════════════════════════════════════════════════════════════════════════
# COVARIANCE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def _build_covariance(signals: list[dict]) -> np.ndarray:
    """Build covariance matrix from volatility and correlation estimates.

    Uses individual volatilities + assumed correlation structure.
    """
    n = len(signals)
    vols = np.array([
        max((s.get("volatility", 1.0) or 1.0) / 100.0, 0.001) * np.sqrt(TRADING_DAYS)
        for s in signals
    ])

    # Build correlation matrix with asset class clustering
    corr = np.eye(n)
    asset_classes = [s.get("asset_class", "other") for s in signals]

    for i in range(n):
        for j in range(i + 1, n):
            if asset_classes[i] == asset_classes[j]:
                # Same asset class: higher correlation
                rho = 0.6 + 0.2 * np.random.RandomState(i * 100 + j).random()
            else:
                # Different asset class: lower correlation
                rho = -0.1 + 0.3 * np.random.RandomState(i * 100 + j).random()
            corr[i, j] = rho
            corr[j, i] = rho

    # Covariance = diag(vol) @ corr @ diag(vol)
    D = np.diag(vols)
    cov = D @ corr @ D

    # Ensure positive semi-definite
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 0:
        cov += np.eye(n) * (abs(eigvals.min()) + 1e-6)

    return cov


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION — Minimum Variance
# ═══════════════════════════════════════════════════════════════════════════════

def _min_variance_portfolio(
    names: list[str],
    cov: np.ndarray,
    expected_returns: np.ndarray,
) -> OptimalPortfolio:
    """Find the minimum variance portfolio using analytical solution.

    w* = Sigma^{-1} @ 1 / (1^T @ Sigma^{-1} @ 1)
    Then clip to [MIN_WEIGHT, MAX_WEIGHT] and renormalize.
    """
    n = len(names)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    ones = np.ones(n)
    raw_weights = inv_cov @ ones
    raw_weights = raw_weights / raw_weights.sum()

    # Clip and renormalize
    weights = np.clip(raw_weights, MIN_WEIGHT, MAX_WEIGHT)
    weights = weights / weights.sum()

    port_return = float(weights @ expected_returns)
    port_vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (port_return - RISK_FREE_RATE) / max(port_vol, 1e-8)

    return OptimalPortfolio(
        name="Minimum Variance",
        weights={names[i]: round(float(weights[i]), 4) for i in range(n)},
        expected_return=round(port_return, 4),
        volatility=round(port_vol, 4),
        sharpe_ratio=round(sharpe, 2),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION — Maximum Sharpe
# ═══════════════════════════════════════════════════════════════════════════════

def _max_sharpe_portfolio(
    names: list[str],
    cov: np.ndarray,
    expected_returns: np.ndarray,
) -> OptimalPortfolio:
    """Find the maximum Sharpe ratio portfolio.

    w* = Sigma^{-1} @ (mu - rf*1) / (1^T @ Sigma^{-1} @ (mu - rf*1))
    """
    n = len(names)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    excess_returns = expected_returns - RISK_FREE_RATE
    raw_weights = inv_cov @ excess_returns

    # If all weights negative, fall back to min variance
    if raw_weights.sum() <= 0:
        raw_weights = np.abs(raw_weights)

    raw_weights = raw_weights / raw_weights.sum()

    # Clip and renormalize
    weights = np.clip(raw_weights, MIN_WEIGHT, MAX_WEIGHT)
    weights = weights / weights.sum()

    port_return = float(weights @ expected_returns)
    port_vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (port_return - RISK_FREE_RATE) / max(port_vol, 1e-8)

    return OptimalPortfolio(
        name="Maximum Sharpe",
        weights={names[i]: round(float(weights[i]), 4) for i in range(n)},
        expected_return=round(port_return, 4),
        volatility=round(port_vol, 4),
        sharpe_ratio=round(sharpe, 2),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EQUAL WEIGHT PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════

def _equal_weight_portfolio(
    names: list[str],
    cov: np.ndarray,
    expected_returns: np.ndarray,
) -> OptimalPortfolio:
    """Compute equal-weight portfolio metrics."""
    n = len(names)
    weights = np.ones(n) / n

    port_return = float(weights @ expected_returns)
    port_vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (port_return - RISK_FREE_RATE) / max(port_vol, 1e-8)

    return OptimalPortfolio(
        name="Equal Weight",
        weights={names[i]: round(float(weights[i]), 4) for i in range(n)},
        expected_return=round(port_return, 4),
        volatility=round(port_vol, 4),
        sharpe_ratio=round(sharpe, 2),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EFFICIENT FRONTIER
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_frontier(
    names: list[str],
    cov: np.ndarray,
    expected_returns: np.ndarray,
    n_points: int = FRONTIER_POINTS,
) -> list[EfficientFrontierPoint]:
    """Compute points along the efficient frontier.

    Varies target return from min to max, solving for minimum variance at each.
    Uses simple interpolation between min-var and max-sharpe weights.
    """
    n = len(names)
    min_ret = float(expected_returns.min())
    max_ret = float(expected_returns.max())

    if abs(max_ret - min_ret) < 1e-6:
        return []

    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    frontier = []
    target_returns = np.linspace(min_ret, max_ret, n_points)

    for target_ret in target_returns:
        # Analytical two-fund theorem
        ones = np.ones(n)
        A = float(ones @ inv_cov @ expected_returns)
        B = float(expected_returns @ inv_cov @ expected_returns)
        C = float(ones @ inv_cov @ ones)
        D = B * C - A * A

        if abs(D) < 1e-10:
            continue

        lam1 = (B - A * target_ret) / D
        lam2 = (C * target_ret - A) / D
        weights = lam1 * (inv_cov @ ones) + lam2 * (inv_cov @ expected_returns)

        # Clip and renormalize
        weights = np.clip(weights, MIN_WEIGHT, MAX_WEIGHT)
        weights = weights / weights.sum()

        port_vol = float(np.sqrt(weights @ cov @ weights))
        actual_ret = float(weights @ expected_returns)
        sharpe = (actual_ret - RISK_FREE_RATE) / max(port_vol, 1e-8)

        frontier.append(EfficientFrontierPoint(
            target_return=round(actual_ret, 4),
            volatility=round(port_vol, 4),
            sharpe_ratio=round(sharpe, 2),
            weights={names[i]: round(float(weights[i]), 4) for i in range(n)},
        ))

    return frontier


# ═══════════════════════════════════════════════════════════════════════════════
# TRACKING ERROR
# ═══════════════════════════════════════════════════════════════════════════════

def _tracking_error(
    current_weights: dict[str, float],
    optimal_weights: dict[str, float],
    names: list[str],
    cov: np.ndarray,
) -> float:
    """Compute tracking error between current and optimal portfolio."""
    n = len(names)
    w_current = np.array([current_weights.get(name, 1.0 / n) for name in names])
    w_optimal = np.array([optimal_weights.get(name, 1.0 / n) for name in names])

    diff = w_current - w_optimal
    te = float(np.sqrt(diff @ cov @ diff))
    return round(te, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT RUN
# ═══════════════════════════════════════════════════════════════════════════════

async def run(
    run_id: str,
    signals: list[dict],
    current_positions: dict | None = None,
) -> StepResult:
    """Execute Portfolio Optimizer: Markowitz mean-variance optimization.

    Args:
        run_id: Pipeline run identifier
        signals: Market signal dicts from market_intelligence
        current_positions: Optional current position weights from position_sizer

    Returns:
        StepResult with OptimizerReport data
    """
    t0 = time.monotonic()
    console.print("[bold cyan]Agent Sentinel — Portfolio Optimizer[/] computing efficient frontier...")

    if not signals or len(signals) < 2:
        latency = (time.monotonic() - t0) * 1000
        console.print("[red]Portfolio Optimizer: need at least 2 assets[/]")
        return StepResult(
            step_name="portfolio_optimization",
            status=StepStatus.FAILED,
            error="Need at least 2 assets for optimization",
            agent_used="portfolio_optimizer",
            latency_ms=latency,
        )

    # === Build inputs ===
    names, expected_returns = _estimate_returns(signals)
    cov = _build_covariance(signals)

    # === Optimize ===
    min_var = _min_variance_portfolio(names, cov, expected_returns)
    max_sharpe = _max_sharpe_portfolio(names, cov, expected_returns)
    equal_wt = _equal_weight_portfolio(names, cov, expected_returns)

    # === Efficient frontier ===
    frontier = _compute_frontier(names, cov, expected_returns)

    # === Tracking error ===
    te = 0.0
    if current_positions:
        current_weights = {}
        for p in current_positions.get("positions", []):
            if isinstance(p, dict):
                current_weights[p.get("symbol", "")] = p.get("final_weight", 0.0)
        if current_weights:
            te = _tracking_error(current_weights, max_sharpe.weights, names, cov)

    report = OptimizerReport(
        min_variance=min_var,
        max_sharpe=max_sharpe,
        equal_weight=equal_wt,
        frontier=frontier,
        asset_names=names,
        expected_returns={names[i]: round(float(expected_returns[i]), 4) for i in range(len(names))},
        tracking_error=te,
    )

    latency = (time.monotonic() - t0) * 1000

    # === Audit ===
    db.save_audit(
        run_id, "portfolio_optimization", "portfolio_optimizer",
        input_summary=f"{len(signals)} assets, rf={RISK_FREE_RATE:.1%}",
        output_summary=(
            f"MinVar Sharpe={min_var.sharpe_ratio:.2f}, "
            f"MaxSharpe Sharpe={max_sharpe.sharpe_ratio:.2f}, "
            f"Frontier={len(frontier)} pts, TE={te:.4f}"
        ),
        model_used="markowitz_mvo",
        latency_ms=latency,
    )

    # === Display ===
    _print_report(report)

    confidence = 80 if len(signals) >= 5 else 60

    console.print(
        f"[green]Portfolio Optimizer done[/] — Sharpe: MinVar={min_var.sharpe_ratio:.2f}, "
        f"MaxSharpe={max_sharpe.sharpe_ratio:.2f}, {len(frontier)} frontier pts "
        f"in {int(latency)}ms"
    )

    return StepResult(
        step_name="portfolio_optimization",
        status=StepStatus.SUCCESS,
        data={
            "min_variance": {
                "weights": min_var.weights,
                "expected_return": min_var.expected_return,
                "volatility": min_var.volatility,
                "sharpe_ratio": min_var.sharpe_ratio,
            },
            "max_sharpe": {
                "weights": max_sharpe.weights,
                "expected_return": max_sharpe.expected_return,
                "volatility": max_sharpe.volatility,
                "sharpe_ratio": max_sharpe.sharpe_ratio,
            },
            "equal_weight": {
                "weights": equal_wt.weights,
                "expected_return": equal_wt.expected_return,
                "volatility": equal_wt.volatility,
                "sharpe_ratio": equal_wt.sharpe_ratio,
            },
            "frontier_points": len(frontier),
            "tracking_error": te,
            "asset_count": len(names),
            "risk_free_rate": RISK_FREE_RATE,
        },
        confidence=confidence,
        agent_used="portfolio_optimizer",
        model_used="markowitz_mvo",
        latency_ms=latency,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def _print_report(report: OptimizerReport) -> None:
    """Display optimization results."""
    # Portfolio comparison table
    table = Table(title="Portfolio Optimization — Markowitz MVO", show_lines=False)
    table.add_column("Portfolio", style="cyan")
    table.add_column("E[Return]", justify="right")
    table.add_column("Volatility", justify="right")
    table.add_column("Sharpe", justify="right")

    for port in [report.min_variance, report.max_sharpe, report.equal_weight]:
        sharpe_color = "green" if port.sharpe_ratio > 0.5 else ("yellow" if port.sharpe_ratio > 0 else "red")
        table.add_row(
            port.name,
            f"{port.expected_return:.2%}",
            f"{port.volatility:.2%}",
            f"[{sharpe_color}]{port.sharpe_ratio:.2f}[/]",
        )

    console.print(table)

    # Weights table
    wt_table = Table(title="Optimal Weights (Max Sharpe)", show_lines=False)
    wt_table.add_column("Asset", style="cyan")
    wt_table.add_column("MinVar %", justify="right")
    wt_table.add_column("MaxSharpe %", justify="right")
    wt_table.add_column("Equal %", justify="right")

    for name in report.asset_names:
        mv_w = report.min_variance.weights.get(name, 0)
        ms_w = report.max_sharpe.weights.get(name, 0)
        eq_w = report.equal_weight.weights.get(name, 0)
        wt_table.add_row(
            name,
            f"{mv_w:.1%}",
            f"[bold]{ms_w:.1%}[/]",
            f"{eq_w:.1%}",
        )

    console.print(wt_table)

    # Summary panel
    best = report.max_sharpe
    console.print(Panel(
        f"[bold]Best Portfolio:[/] {best.name} (Sharpe {best.sharpe_ratio:.2f})\n"
        f"[bold]Expected Return:[/] {best.expected_return:.2%} annual\n"
        f"[bold]Volatility:[/] {best.volatility:.2%} annual\n"
        f"[bold]Frontier Points:[/] {len(report.frontier)}\n"
        f"[bold]Tracking Error:[/] {report.tracking_error:.4f}\n"
        f"[bold]Risk-Free Rate:[/] {RISK_FREE_RATE:.1%}",
        title="[bold]Portfolio Optimizer — Summary[/]",
        border_style="cyan",
    ))

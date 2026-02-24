"""Agent Sentinel — Strategy Backtester: simulation Monte Carlo des strategies de couverture.

Recoit les strategies generees par consensus_strategy et simule un backtest 30 jours:
- Monte Carlo (50 trajectoires) avec GBM (Geometric Brownian Motion)
- Profils de simulation adaptes par type de strategy (Conservative/Moderate/Aggressive)
- Metriques: Sharpe Ratio, Max Drawdown, Win Rate, P&L simule, Calmar Ratio
- Grade par strategy (A/B/C/D/F base sur le Sharpe)
- Enrichissement Airia best-effort via bridge.execute_backtesting()

Utilise numpy pour les calculs de simulation.
"""

from __future__ import annotations

import json
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
# CONFIGURATION — Parametres de simulation
# ═══════════════════════════════════════════════════════════════════════════════

SIMULATION_DAYS = 30                # Horizon de backtest
DAILY_STEPS = 24                    # Pas par jour (hourly)
TOTAL_STEPS = SIMULATION_DAYS * DAILY_STEPS
RISK_FREE_RATE = 0.04               # 4% annualise (normalise daily)
ANNUALIZATION_FACTOR = np.sqrt(365) # Pour annualiser le Sharpe
NUM_SIMULATIONS = 50                # Monte Carlo: nombre de trajectoires par strategy
INITIAL_CAPITAL = 1_000_000.0       # Capital initial simule (USD)

# Mapping strategy type -> parametres de simulation
STRATEGY_PROFILES = {
    "conservative": {"drift": 0.0002, "vol": 0.008, "hedge_efficiency": 0.85},
    "moderate":     {"drift": 0.0004, "vol": 0.015, "hedge_efficiency": 0.60},
    "aggressive":   {"drift": 0.0008, "vol": 0.025, "hedge_efficiency": 0.30},
}
DEFAULT_PROFILE = {"drift": 0.0004, "vol": 0.018, "hedge_efficiency": 0.50}

# Grade mapping par Sharpe Ratio
SHARPE_GRADES = [
    (2.0, "A"),   # Excellent
    (1.5, "A-"),  # Tres bon
    (1.0, "B+"),  # Bon
    (0.5, "B"),   # Correct
    (0.0, "C"),   # Mediocre
    (-0.5, "D"),  # Mauvais
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestMetrics:
    """Metriques de backtest pour une strategy."""
    strategy_name: str
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate_pct: float = 0.0
    pnl_30d_pct: float = 0.0
    pnl_30d_usd: float = 0.0
    avg_daily_return_pct: float = 0.0
    volatility_annualized_pct: float = 0.0
    calmar_ratio: float = 0.0        # Return / Max Drawdown
    total_trades: int = 0
    grade: str = "C"
    confidence: float = 0.0
    cost_impact_pct: float = 0.0      # Impact du cout de la strategie


@dataclass
class BacktestReport:
    """Rapport complet de backtesting multi-strategy."""
    metrics: list[BacktestMetrics] = field(default_factory=list)
    best_strategy: str = ""
    best_sharpe: float = 0.0
    simulation_days: int = SIMULATION_DAYS
    num_simulations: int = NUM_SIMULATIONS
    airia_enriched: bool = False
    models_used: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _get_strategy_profile(strategy_name: str) -> dict[str, float]:
    """Retourne le profil de simulation associe au nom de la strategy."""
    name_lower = strategy_name.lower()
    for key, profile in STRATEGY_PROFILES.items():
        if key in name_lower:
            return profile
    return DEFAULT_PROFILE


def _simulate_equity_curve(
    drift: float,
    vol: float,
    hedge_efficiency: float,
    cost_pct: float,
    risk_reduction_pct: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simule une courbe d'equity sur TOTAL_STEPS pas via GBM.

    Modele: Geometric Brownian Motion avec ajustement pour hedge.
    - Le drift est ajuste par l'efficacite du hedge
    - La volatilite est reduite proportionnellement au risk_reduction
    - Le cout de la strategy est deduit lineairement

    Args:
        drift: Drift journalier (mu)
        vol: Volatilite journaliere (sigma)
        hedge_efficiency: Efficacite du hedge (0-1)
        cost_pct: Cout de la strategy en % du portefeuille
        risk_reduction_pct: Reduction de risque attendue en %
        rng: Generateur aleatoire numpy

    Returns:
        Courbe d'equity normalisee (commence a 1.0)
    """
    # Ajuster la volatilite par le hedge
    effective_vol = vol * (1 - risk_reduction_pct / 100 * hedge_efficiency)
    effective_vol = max(effective_vol, 0.001)  # Plancher

    # Drift ajuste: drift net - cout lineaire reparti
    daily_cost = cost_pct / 100 / SIMULATION_DAYS
    effective_drift = drift * (1 + hedge_efficiency * 0.3) - daily_cost / DAILY_STEPS

    # Generer les rendements log-normaux
    dt = 1.0 / DAILY_STEPS
    noise = rng.standard_normal(TOTAL_STEPS)
    log_returns = (effective_drift - 0.5 * effective_vol**2) * dt + effective_vol * np.sqrt(dt) * noise

    # Courbe d'equity cumulative
    equity = np.exp(np.cumsum(log_returns))
    equity = np.insert(equity, 0, 1.0)  # Commence a 1.0

    return equity


def _compute_metrics_from_equity(
    equity: np.ndarray,
    strategy_name: str,
    cost_pct: float,
) -> BacktestMetrics:
    """Calcule les metriques de backtest a partir d'une courbe d'equity.

    Args:
        equity: Courbe d'equity normalisee
        strategy_name: Nom de la strategy
        cost_pct: Cout de la strategy

    Returns:
        BacktestMetrics
    """
    # Rendements quotidiens (agreger les hourly)
    daily_equity = equity[::DAILY_STEPS]
    daily_returns = np.diff(daily_equity) / daily_equity[:-1]

    # Sharpe Ratio (annualise)
    daily_rf = RISK_FREE_RATE / 365
    excess_returns = daily_returns - daily_rf
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)

    sharpe = 0.0
    if std_excess > 1e-10:
        sharpe = float(mean_excess / std_excess * ANNUALIZATION_FACTOR)

    # Max Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = float(np.min(drawdown)) * 100  # En %

    # Win Rate (% de jours positifs)
    win_rate = float(np.mean(daily_returns > 0) * 100)

    # P&L 30d
    pnl_pct = float((equity[-1] / equity[0] - 1) * 100)
    pnl_usd = pnl_pct / 100 * INITIAL_CAPITAL

    # Volatilite annualisee
    vol_annual = float(np.std(daily_returns) * ANNUALIZATION_FACTOR * 100)

    # Avg daily return
    avg_daily = float(np.mean(daily_returns) * 100)

    # Calmar ratio
    calmar = 0.0
    if abs(max_dd) > 0.01:
        calmar = float(pnl_pct / abs(max_dd))

    # Nombre de "trades" simules (jours actifs)
    total_trades = len(daily_returns)

    # Grade
    grade = _sharpe_to_grade(sharpe)

    # Confidence: base sur la stabilite des rendements (inverse du CV)
    if abs(np.mean(daily_returns)) > 1e-10:
        cv = abs(np.std(daily_returns) / np.mean(daily_returns))
        confidence = max(10, min(95, 90 - cv * 10))
    else:
        confidence = 30.0

    return BacktestMetrics(
        strategy_name=strategy_name,
        sharpe_ratio=round(sharpe, 3),
        max_drawdown_pct=round(max_dd, 2),
        win_rate_pct=round(win_rate, 1),
        pnl_30d_pct=round(pnl_pct, 2),
        pnl_30d_usd=round(pnl_usd, 0),
        avg_daily_return_pct=round(avg_daily, 4),
        volatility_annualized_pct=round(vol_annual, 2),
        calmar_ratio=round(calmar, 3),
        total_trades=total_trades,
        grade=grade,
        confidence=round(confidence, 1),
        cost_impact_pct=cost_pct,
    )


def _sharpe_to_grade(sharpe: float) -> str:
    """Convertit un Sharpe Ratio en grade lettre."""
    for threshold, grade in SHARPE_GRADES:
        if sharpe >= threshold:
            return grade
    return "F"


def _run_monte_carlo(
    strategy: dict,
    num_sims: int = NUM_SIMULATIONS,
    seed: int | None = None,
) -> BacktestMetrics:
    """Execute un Monte Carlo backtest pour une strategy.

    Lance num_sims simulations et retourne les metriques medianes.

    Args:
        strategy: Dict de la strategy (name, cost_estimate_pct, risk_reduction_pct, ...)
        num_sims: Nombre de simulations
        seed: Seed aleatoire (reproductibilite)

    Returns:
        BacktestMetrics avec les valeurs medianes
    """
    rng = np.random.default_rng(seed)
    profile = _get_strategy_profile(strategy.get("name", ""))

    cost_pct = strategy.get("cost_estimate_pct", 0.3)
    risk_reduction = strategy.get("risk_reduction_pct", 50)

    all_sharpes: list[float] = []
    all_max_dd: list[float] = []
    all_win_rates: list[float] = []
    all_pnl: list[float] = []
    all_pnl_usd: list[float] = []
    all_vol: list[float] = []
    all_calmar: list[float] = []

    for _ in range(num_sims):
        equity = _simulate_equity_curve(
            drift=profile["drift"],
            vol=profile["vol"],
            hedge_efficiency=profile["hedge_efficiency"],
            cost_pct=cost_pct,
            risk_reduction_pct=risk_reduction,
            rng=rng,
        )
        m = _compute_metrics_from_equity(equity, strategy.get("name", ""), cost_pct)
        all_sharpes.append(m.sharpe_ratio)
        all_max_dd.append(m.max_drawdown_pct)
        all_win_rates.append(m.win_rate_pct)
        all_pnl.append(m.pnl_30d_pct)
        all_pnl_usd.append(m.pnl_30d_usd)
        all_vol.append(m.volatility_annualized_pct)
        all_calmar.append(m.calmar_ratio)

    # Valeurs medianes (plus robuste que la moyenne)
    median_sharpe = float(np.median(all_sharpes))
    median_dd = float(np.median(all_max_dd))
    median_wr = float(np.median(all_win_rates))
    median_pnl = float(np.median(all_pnl))
    median_pnl_usd = float(np.median(all_pnl_usd))
    median_vol = float(np.median(all_vol))
    median_calmar = float(np.median(all_calmar))

    grade = _sharpe_to_grade(median_sharpe)

    # Confidence: base sur la dispersion des Sharpe (IQR)
    q25, q75 = np.percentile(all_sharpes, [25, 75])
    iqr = q75 - q25
    confidence = max(20, min(95, 85 - iqr * 15))

    return BacktestMetrics(
        strategy_name=strategy.get("name", "Unknown"),
        sharpe_ratio=round(median_sharpe, 3),
        max_drawdown_pct=round(median_dd, 2),
        win_rate_pct=round(median_wr, 1),
        pnl_30d_pct=round(median_pnl, 2),
        pnl_30d_usd=round(median_pnl_usd, 0),
        avg_daily_return_pct=round(median_pnl / SIMULATION_DAYS, 4),
        volatility_annualized_pct=round(median_vol, 2),
        calmar_ratio=round(median_calmar, 3),
        total_trades=SIMULATION_DAYS,
        grade=grade,
        confidence=round(confidence, 1),
        cost_impact_pct=cost_pct,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AIRIA ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _enrich_with_airia(
    strategies: list[dict],
    local_metrics: list[BacktestMetrics],
) -> dict[str, Any] | None:
    """Enrichissement via Airia — analyse qualitative des backtests (best-effort).

    Envoie les strategies et resultats locaux a Airia pour obtenir
    une analyse narrative et des recommandations supplementaires.

    Returns:
        Dict avec analyse Airia ou None si indisponible
    """
    if not bridge.is_available:
        return None

    try:
        input_data = [
            {
                "strategy": s.get("name", ""),
                "cost_pct": s.get("cost_estimate_pct", 0),
                "risk_reduction_pct": s.get("risk_reduction_pct", 0),
                "backtest_sharpe": m.sharpe_ratio,
                "backtest_max_dd": m.max_drawdown_pct,
                "backtest_pnl": m.pnl_30d_pct,
                "backtest_grade": m.grade,
            }
            for s, m in zip(strategies, local_metrics)
        ]

        result = bridge.execute_backtesting(input_data)

        if result.get("ok"):
            console.print(f"  [dim]Airia backtesting: {result.get('latency_ms', 0)}ms[/]")
            return result.get("parsed") or result.get("result")

        console.print(f"  [yellow]Airia backtesting failed: {result.get('error', '?')}[/]")
    except Exception as e:
        console.print(f"  [yellow]Airia backtesting error: {e}[/]")

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT RUN — Point d'entree principal
# ═══════════════════════════════════════════════════════════════════════════════

async def run(
    run_id: str,
    strategies: list[dict],
) -> StepResult:
    """Execute l'agent Strategy Backtester: simulation Monte Carlo + metriques.

    Args:
        run_id: Identifiant du run pipeline
        strategies: Liste de strategies (dicts avec name, description, instruments,
                    cost_estimate_pct, risk_reduction_pct, confidence, rationale)

    Returns:
        StepResult avec BacktestReport dans data
    """
    t0 = time.monotonic()
    console.print("[bold cyan]Agent Sentinel — Strategy Backtester[/] simulation en cours...")

    if not strategies:
        latency = (time.monotonic() - t0) * 1000
        console.print("[yellow]Backtester: aucune strategy a tester[/]")
        return StepResult(
            step_name="backtesting",
            status=StepStatus.FAILED,
            error="Aucune strategy fournie",
            agent_used="backtester",
            latency_ms=latency,
        )

    models_used = ["numpy-monte-carlo"]
    all_metrics: list[BacktestMetrics] = []

    # === Etape 1: Monte Carlo backtest par strategy ===
    for i, strategy in enumerate(strategies):
        # Support both dict and dataclass-style objects
        if not isinstance(strategy, dict):
            strategy = strategy.model_dump(mode="json") if hasattr(strategy, "model_dump") else {"name": str(strategy)}

        name = strategy.get("name", f"Strategy-{i+1}")
        console.print(f"  [dim]Backtesting {name}: {NUM_SIMULATIONS} simulations x {SIMULATION_DAYS} jours...[/]")

        # Seed reproductible par strategy (hash du nom + run_id)
        seed = hash(f"{run_id}-{name}") % (2**31)
        metrics = _run_monte_carlo(strategy, num_sims=NUM_SIMULATIONS, seed=seed)
        all_metrics.append(metrics)

        console.print(
            f"    Sharpe={metrics.sharpe_ratio:.2f}, "
            f"MaxDD={metrics.max_drawdown_pct:.1f}%, "
            f"WinRate={metrics.win_rate_pct:.0f}%, "
            f"P&L={metrics.pnl_30d_pct:+.1f}% -> [{_grade_color(metrics.grade)}]{metrics.grade}[/]"
        )

    # === Etape 2: Enrichissement Airia (best-effort) ===
    # Ensure we pass dicts for Airia
    strategy_dicts = []
    for s in strategies:
        if isinstance(s, dict):
            strategy_dicts.append(s)
        elif hasattr(s, "model_dump"):
            strategy_dicts.append(s.model_dump(mode="json"))
        else:
            strategy_dicts.append({"name": str(s)})

    airia_result = _enrich_with_airia(strategy_dicts, all_metrics)
    airia_enriched = airia_result is not None
    if airia_enriched:
        models_used.append("airia")

        # Si Airia retourne des ajustements de grade, les integrer
        if isinstance(airia_result, dict):
            for bt in airia_result.get("backtests", []):
                matching = next(
                    (m for m in all_metrics if m.strategy_name.lower() == bt.get("strategy", "").lower()),
                    None,
                )
                if matching and bt.get("grade"):
                    # Prendre le grade le plus conservateur (plus mauvais)
                    airia_grade = bt["grade"]
                    if _grade_rank(airia_grade) > _grade_rank(matching.grade):
                        matching.grade = airia_grade

    # === Etape 3: Identifier la meilleure strategy ===
    best_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i].sharpe_ratio)
    best_metrics = all_metrics[best_idx]

    report = BacktestReport(
        metrics=all_metrics,
        best_strategy=best_metrics.strategy_name,
        best_sharpe=best_metrics.sharpe_ratio,
        simulation_days=SIMULATION_DAYS,
        num_simulations=NUM_SIMULATIONS,
        airia_enriched=airia_enriched,
        models_used=models_used,
    )

    latency = (time.monotonic() - t0) * 1000

    # === Audit DB ===
    db.save_audit(
        run_id, "backtesting", "backtester",
        input_summary=f"{len(strategies)} strategies, {NUM_SIMULATIONS} sims x {SIMULATION_DAYS}j",
        output_summary=(
            f"best={best_metrics.strategy_name} (Sharpe={best_metrics.sharpe_ratio:.2f}, "
            f"grade={best_metrics.grade}), airia={'OUI' if airia_enriched else 'NON'}"
        ),
        model_used=", ".join(models_used),
        latency_ms=latency,
    )

    # === Affichage ===
    _print_backtest_report(report)

    # Confidence globale: moyenne ponderee des confidences individuelles
    avg_confidence = sum(m.confidence for m in all_metrics) / len(all_metrics) if all_metrics else 0

    console.print(
        f"[green]Backtester done[/] — best: {best_metrics.strategy_name} "
        f"(Sharpe={best_metrics.sharpe_ratio:.2f}, {best_metrics.grade}), "
        f"{len(strategies)} strategies in {int(latency)}ms"
    )

    return StepResult(
        step_name="backtesting",
        status=StepStatus.SUCCESS,
        data={
            "backtests": [
                {
                    "strategy": m.strategy_name,
                    "sharpe_ratio": m.sharpe_ratio,
                    "max_drawdown_pct": m.max_drawdown_pct,
                    "win_rate_pct": m.win_rate_pct,
                    "pnl_30d_pct": m.pnl_30d_pct,
                    "pnl_30d_usd": m.pnl_30d_usd,
                    "avg_daily_return_pct": m.avg_daily_return_pct,
                    "volatility_annualized_pct": m.volatility_annualized_pct,
                    "calmar_ratio": m.calmar_ratio,
                    "total_trades": m.total_trades,
                    "grade": m.grade,
                    "confidence": m.confidence,
                    "cost_impact_pct": m.cost_impact_pct,
                }
                for m in all_metrics
            ],
            "best_strategy": report.best_strategy,
            "best_sharpe": report.best_sharpe,
            "simulation_days": report.simulation_days,
            "num_simulations": report.num_simulations,
            "airia_enriched": report.airia_enriched,
            "models_used": report.models_used,
        },
        confidence=round(avg_confidence, 1),
        agent_used="backtester",
        model_used=", ".join(models_used),
        latency_ms=latency,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _grade_rank(grade: str) -> int:
    """Convertit un grade en rang numerique (A=1 meilleur, F=7 pire)."""
    ranks = {"A": 1, "A-": 2, "B+": 3, "B": 4, "C": 5, "D": 6, "F": 7}
    return ranks.get(grade, 5)


def _grade_color(grade: str) -> str:
    """Retourne la couleur Rich pour un grade."""
    if grade in ("A", "A-"):
        return "bold green"
    if grade in ("B+", "B"):
        return "green"
    if grade == "C":
        return "yellow"
    if grade == "D":
        return "red"
    return "bold red"


# ═══════════════════════════════════════════════════════════════════════════════
# AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _print_backtest_report(report: BacktestReport) -> None:
    """Affiche le rapport de backtesting complet."""
    table = Table(
        title=f"Strategy Backtesting — {report.simulation_days}j, {report.num_simulations} simulations",
        show_lines=True,
    )
    table.add_column("Strategy", style="cyan")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("P&L 30d", justify="right")
    table.add_column("Vol Ann.", justify="right")
    table.add_column("Calmar", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Grade", justify="center")

    for m in report.metrics:
        is_best = m.strategy_name == report.best_strategy
        name = f"[bold]{m.strategy_name}[/] *" if is_best else m.strategy_name

        sharpe_color = "green" if m.sharpe_ratio > 1.0 else ("yellow" if m.sharpe_ratio > 0 else "red")
        dd_color = "green" if m.max_drawdown_pct > -5 else ("yellow" if m.max_drawdown_pct > -10 else "red")
        pnl_color = "green" if m.pnl_30d_pct > 0 else "red"
        g_color = _grade_color(m.grade)

        table.add_row(
            name,
            f"[{sharpe_color}]{m.sharpe_ratio:.2f}[/]",
            f"[{dd_color}]{m.max_drawdown_pct:.1f}%[/]",
            f"{m.win_rate_pct:.0f}%",
            f"[{pnl_color}]{m.pnl_30d_pct:+.1f}%[/]",
            f"{m.volatility_annualized_pct:.1f}%",
            f"{m.calmar_ratio:.2f}",
            f"{m.cost_impact_pct:.2f}%",
            f"[{g_color}]{m.grade}[/]",
        )

    console.print(table)

    # Panel de synthese
    best = next((m for m in report.metrics if m.strategy_name == report.best_strategy), None)
    if best:
        best_color = _grade_color(best.grade).replace("bold ", "")
        console.print(Panel(
            f"[bold]Best Strategy:[/] [{best_color}]{best.strategy_name}[/]\n"
            f"[bold]Sharpe Ratio:[/] {best.sharpe_ratio:.2f}\n"
            f"[bold]P&L 30d:[/] {best.pnl_30d_pct:+.1f}% (${best.pnl_30d_usd:+,.0f})\n"
            f"[bold]Max Drawdown:[/] {best.max_drawdown_pct:.1f}%\n"
            f"[bold]Grade:[/] [{_grade_color(best.grade)}]{best.grade}[/]\n"
            f"[bold]Airia Enriched:[/] {'OUI' if report.airia_enriched else 'NON'}\n"
            f"[bold]Modeles:[/] {', '.join(report.models_used)}",
            title="[bold]Backtester — Synthese[/]",
            border_style=best_color,
        ))

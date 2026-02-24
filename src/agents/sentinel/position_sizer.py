"""Agent Sentinel — Position Sizer: dimensionnement optimal des positions.

Implemente:
- Kelly Criterion pour le sizing optimal de chaque bet
- Risk Parity pour l'allocation inter-actifs
- Limite de risque par trade (defaut 2% du compte)
- Ajustement pour la correlation entre actifs

Utilise numpy pour les calculs matriciels.
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
# CONFIGURATION — Parametres de risque
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_RISK_PER_TRADE = 0.02   # 2% du compte par trade
MAX_PORTFOLIO_RISK = 0.10       # 10% risque max total du portefeuille
KELLY_FRACTION = 0.5            # Demi-Kelly (plus conservateur)
MIN_POSITION_SIZE = 0.001       # Taille minimum d'une position (0.1%)
MAX_SINGLE_POSITION = 0.15      # Taille max d'une position (15% du portefeuille)
DEFAULT_LEVERAGE = 10           # Levier par defaut (MEXC Futures)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PositionSize:
    """Taille de position calculee pour un actif."""
    symbol: str
    kelly_fraction: float = 0.0       # Fraction Kelly brute
    kelly_adjusted: float = 0.0       # Fraction Kelly ajustee (half-Kelly)
    risk_parity_weight: float = 0.0   # Poids Risk Parity
    final_weight: float = 0.0         # Poids final apres ajustements
    position_usd: float = 0.0         # Taille en USD
    max_loss_usd: float = 0.0         # Perte max estimee
    leverage: int = DEFAULT_LEVERAGE
    method: str = "kelly+risk_parity"


@dataclass
class SizingResult:
    """Resultat complet du dimensionnement de portefeuille."""
    positions: list[PositionSize] = field(default_factory=list)
    total_allocated_pct: float = 0.0
    total_risk_pct: float = 0.0
    account_balance: float = 0.0
    risk_per_trade: float = DEFAULT_RISK_PER_TRADE
    correlation_adjustment: float = 1.0  # Facteur d'ajustement correlation


# ═══════════════════════════════════════════════════════════════════════════════
# KELLY CRITERION
# ═══════════════════════════════════════════════════════════════════════════════

def _kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """Calcule la fraction Kelly optimale.

    f* = (p * b - q) / b
    ou: p = probabilite de gain, b = ratio gain/perte, q = 1 - p

    Args:
        win_rate: Probabilite de gain (0-1)
        avg_win: Gain moyen (en %)
        avg_loss: Perte moyenne (en %, valeur positive)

    Returns:
        Fraction Kelly (0 a 1), plafonnee
    """
    if avg_loss <= 0 or win_rate <= 0:
        return 0.0

    b = avg_win / avg_loss  # Ratio gain/perte
    p = win_rate
    q = 1 - p

    kelly = (p * b - q) / b

    # Plafonner entre 0 et 1
    return max(0.0, min(1.0, kelly))


def _compute_kelly_fractions(risk_profile: dict) -> dict[str, float]:
    """Calcule la fraction Kelly pour chaque actif base sur le profil de risque.

    Estime le win_rate et les ratios gain/perte a partir des risk_scores
    et des niveaux d'alerte.
    """
    fractions: dict[str, float] = {}
    risk_by_asset = risk_profile.get("risk_by_asset_class", {})
    overall_risk = risk_profile.get("overall_risk", 50)

    for asset_class, risk_score in risk_by_asset.items():
        # Estimation du win rate: inversement proportionnel au risque
        # Risque 0 → win_rate ~0.65, Risque 100 → win_rate ~0.35
        win_rate = 0.65 - (risk_score / 100) * 0.30

        # Ratio gain/perte base sur les thresholds de config
        # TP 0.4%, SL 0.25% → ratio = 1.6
        avg_win = 0.4   # TP par defaut (config.trading)
        avg_loss = 0.25  # SL par defaut

        kelly = _kelly_criterion(win_rate, avg_win, avg_loss)
        fractions[asset_class] = round(kelly * KELLY_FRACTION, 4)  # Half-Kelly

    return fractions


# ═══════════════════════════════════════════════════════════════════════════════
# RISK PARITY
# ═══════════════════════════════════════════════════════════════════════════════

def _risk_parity_weights(risk_scores: dict[str, float]) -> dict[str, float]:
    """Calcule les poids Risk Parity: allouer inversement proportionnel au risque.

    Les actifs moins risques recoivent plus de poids pour equaliser
    la contribution au risque du portefeuille.
    """
    if not risk_scores:
        return {}

    # Inverser les scores: moins de risque = plus de poids
    inverse_risks = {}
    for asset, risk in risk_scores.items():
        # Eviter division par zero: risque minimum = 5
        inverse_risks[asset] = 1.0 / max(risk, 5)

    # Normaliser pour que la somme = 1
    total = sum(inverse_risks.values())
    if total <= 0:
        return {asset: 1.0 / len(risk_scores) for asset in risk_scores}

    return {
        asset: round(inv / total, 4)
        for asset, inv in inverse_risks.items()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CORRELATION ADJUSTMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _correlation_adjustment(risk_profile: dict) -> float:
    """Ajuste la taille des positions en fonction de la correlation inter-actifs.

    Si une cascade est detectee (actifs correles), reduire les positions
    car le risque effectif est plus eleve.

    Returns:
        Facteur multiplicatif (0.5 a 1.0)
    """
    cascade = risk_profile.get("cascade_detected", False)
    overall_risk = risk_profile.get("overall_risk", 50)

    if cascade:
        # Cascade detectee: reduire significativement
        if overall_risk > 70:
            return 0.5   # Reduire de moitie
        return 0.65       # Reduire d'un tiers

    if overall_risk > 60:
        return 0.75       # Reduire de 25% en conditions difficiles

    return 1.0            # Pas d'ajustement


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING — Combinaison Kelly + Risk Parity
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_positions(
    risk_profile: dict,
    account_balance: float,
    risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
) -> SizingResult:
    """Calcule la taille optimale de chaque position.

    Combine Kelly Criterion et Risk Parity, ajuste pour la correlation.
    """
    risk_by_class = risk_profile.get("risk_by_asset_class", {})

    # Etape 1: Kelly fractions
    kelly_fractions = _compute_kelly_fractions(risk_profile)

    # Etape 2: Risk Parity weights
    parity_weights = _risk_parity_weights(risk_by_class)

    # Etape 3: Ajustement correlation
    corr_adj = _correlation_adjustment(risk_profile)

    # Etape 4: Combinaison (50% Kelly, 50% Risk Parity)
    positions: list[PositionSize] = []
    total_allocated = 0.0
    total_risk = 0.0

    for asset_class in risk_by_class:
        kelly = kelly_fractions.get(asset_class, 0.0)
        parity = parity_weights.get(asset_class, 0.0)

        # Poids combine: moyenne Kelly + Parity, ajuste pour correlation
        combined = (kelly * 0.5 + parity * 0.5) * corr_adj

        # Appliquer les limites
        combined = max(MIN_POSITION_SIZE, min(MAX_SINGLE_POSITION, combined))

        # Verifier que le risque total ne depasse pas le max
        risk_contribution = combined * (risk_by_class.get(asset_class, 50) / 100)
        if total_risk + risk_contribution > MAX_PORTFOLIO_RISK:
            # Reduire pour respecter la limite
            max_allowed = MAX_PORTFOLIO_RISK - total_risk
            if max_allowed > 0:
                combined = max_allowed / (risk_by_class.get(asset_class, 50) / 100 + 1e-10)
                combined = max(MIN_POSITION_SIZE, min(combined, MAX_SINGLE_POSITION))
            else:
                combined = 0.0

        # Calculer en USD
        position_usd = round(account_balance * combined, 2)
        max_loss = round(account_balance * risk_per_trade, 2)

        positions.append(PositionSize(
            symbol=asset_class,
            kelly_fraction=round(kelly, 4),
            kelly_adjusted=round(kelly * KELLY_FRACTION, 4),
            risk_parity_weight=round(parity, 4),
            final_weight=round(combined, 4),
            position_usd=position_usd,
            max_loss_usd=max_loss,
        ))

        total_allocated += combined
        total_risk += risk_contribution

    return SizingResult(
        positions=positions,
        total_allocated_pct=round(total_allocated * 100, 2),
        total_risk_pct=round(total_risk * 100, 2),
        account_balance=account_balance,
        risk_per_trade=risk_per_trade,
        correlation_adjustment=corr_adj,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT RUN — Point d'entree principal
# ═══════════════════════════════════════════════════════════════════════════════

async def run(
    run_id: str,
    risk_profile: dict,
    account_balance: float = 10_000.0,
    risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
) -> StepResult:
    """Execute l'agent Position Sizer: calcul Kelly + Risk Parity.

    Args:
        run_id: Identifiant du run pipeline
        risk_profile: Profil de risque du risk_aggregator (overall_risk, risk_by_asset_class, cascade_detected)
        account_balance: Solde du compte en USD
        risk_per_trade: Risque maximum par trade (defaut 2%)

    Returns:
        StepResult avec SizingResult dans data
    """
    t0 = time.monotonic()
    console.print("[bold cyan]Agent Sentinel — Position Sizer[/] calcul en cours...")

    # Validation des inputs
    if not risk_profile or not risk_profile.get("risk_by_asset_class"):
        latency = (time.monotonic() - t0) * 1000
        console.print("[red]Position Sizer: profil de risque vide[/]")
        return StepResult(
            step_name="position_sizing",
            status=StepStatus.FAILED,
            error="Profil de risque vide ou invalide",
            agent_used="position_sizer",
            latency_ms=latency,
        )

    if account_balance <= 0:
        latency = (time.monotonic() - t0) * 1000
        console.print("[red]Position Sizer: solde du compte invalide[/]")
        return StepResult(
            step_name="position_sizing",
            status=StepStatus.FAILED,
            error=f"Solde du compte invalide: {account_balance}",
            agent_used="position_sizer",
            latency_ms=latency,
        )

    # === Calcul des positions ===
    result = _compute_positions(risk_profile, account_balance, risk_per_trade)

    latency = (time.monotonic() - t0) * 1000

    # === Audit DB ===
    db.save_audit(
        run_id, "position_sizing", "position_sizer",
        input_summary=(
            f"balance=${account_balance:,.0f}, risk/trade={risk_per_trade:.1%}, "
            f"risque_global={risk_profile.get('overall_risk', 0):.0f}"
        ),
        output_summary=(
            f"{len(result.positions)} positions, alloc={result.total_allocated_pct:.1f}%, "
            f"risque={result.total_risk_pct:.1f}%, corr_adj={result.correlation_adjustment:.2f}"
        ),
        model_used="kelly+risk_parity",
        latency_ms=latency,
    )

    # === Affichage ===
    _print_sizing_result(result)

    confidence = 85 if not risk_profile.get("cascade_detected") else 60

    console.print(
        f"[green]Position Sizer done[/] — {len(result.positions)} positions, "
        f"alloc={result.total_allocated_pct:.1f}%, risque={result.total_risk_pct:.1f}% "
        f"in {int(latency)}ms"
    )

    return StepResult(
        step_name="position_sizing",
        status=StepStatus.SUCCESS,
        data={
            "positions": [
                {
                    "symbol": p.symbol,
                    "kelly_fraction": p.kelly_fraction,
                    "kelly_adjusted": p.kelly_adjusted,
                    "risk_parity_weight": p.risk_parity_weight,
                    "final_weight": p.final_weight,
                    "position_usd": p.position_usd,
                    "max_loss_usd": p.max_loss_usd,
                    "leverage": p.leverage,
                    "method": p.method,
                }
                for p in result.positions
            ],
            "total_allocated_pct": result.total_allocated_pct,
            "total_risk_pct": result.total_risk_pct,
            "account_balance": result.account_balance,
            "risk_per_trade": result.risk_per_trade,
            "correlation_adjustment": result.correlation_adjustment,
        },
        confidence=confidence,
        agent_used="position_sizer",
        model_used="kelly+risk_parity",
        latency_ms=latency,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _print_sizing_result(result: SizingResult) -> None:
    """Affiche le resultat du dimensionnement des positions."""
    table = Table(title=f"Position Sizing — Balance: ${result.account_balance:,.0f}", show_lines=False)
    table.add_column("Asset Class", style="cyan")
    table.add_column("Kelly", justify="right")
    table.add_column("Risk Parity", justify="right")
    table.add_column("Final %", justify="right")
    table.add_column("Position USD", justify="right")
    table.add_column("Max Loss", justify="right")
    table.add_column("Leverage", justify="center")

    for p in result.positions:
        weight_color = "green" if p.final_weight < 0.05 else ("yellow" if p.final_weight < 0.10 else "red")
        table.add_row(
            p.symbol,
            f"{p.kelly_adjusted:.2%}",
            f"{p.risk_parity_weight:.2%}",
            f"[{weight_color}]{p.final_weight:.2%}[/]",
            f"${p.position_usd:,.0f}",
            f"${p.max_loss_usd:,.0f}",
            f"{p.leverage}x",
        )

    console.print(table)

    # Panel de synthese
    risk_color = "green" if result.total_risk_pct < 5 else ("yellow" if result.total_risk_pct < 8 else "red")
    console.print(Panel(
        f"[bold]Allocation totale:[/] {result.total_allocated_pct:.1f}%\n"
        f"[bold]Risque total:[/] [{risk_color}]{result.total_risk_pct:.1f}%[/] (max {MAX_PORTFOLIO_RISK*100:.0f}%)\n"
        f"[bold]Risque par trade:[/] {result.risk_per_trade:.1%}\n"
        f"[bold]Ajustement correlation:[/] {result.correlation_adjustment:.2f}x\n"
        f"[bold]Methode:[/] Kelly ({KELLY_FRACTION:.0%}) + Risk Parity",
        title="[bold]Position Sizer — Synthese[/]",
        border_style=risk_color,
    ))

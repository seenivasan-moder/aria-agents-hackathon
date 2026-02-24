"""Agent Sentinel — Sentiment Scorer: scoring multi-facteur du sentiment marche.

Analyse les signaux de marche pour en extraire un score de sentiment composite:
- Momentum (direction + amplitude des variations)
- Mean Reversion (detection de retour a la moyenne)
- Regime de volatilite (low/normal/high/extreme)

Score composite: -100 (extreme bearish) a +100 (extreme bullish).

Utilise Ollama OL1 (qwen3:1.7b) pour classification rapide.
Fallback rule-based si IA indisponible.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.pipeline_engine import StepResult, StepStatus
from src.config import config
from src.services.lm_cluster import query_ollama
from src import database as db

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SENTIMENT_RANGE = (-100, 100)       # Plage du score de sentiment
MOMENTUM_WEIGHT = 0.40              # Poids du momentum dans le score composite
MEAN_REVERSION_WEIGHT = 0.25        # Poids de la mean reversion
VOLATILITY_WEIGHT = 0.20            # Poids du regime de volatilite
AI_SENTIMENT_WEIGHT = 0.15          # Poids de la classification IA

# Seuils de volatilite
VOL_LOW = 1.0       # < 1% = basse volatilite
VOL_NORMAL = 3.0    # 1-3% = normale
VOL_HIGH = 6.0      # 3-6% = haute
# > 6% = extreme

OL1_SENTIMENT_PROMPT = """\
Classe le sentiment de marche de ces actifs. Pour chaque actif, donne un score entre -100 (tres bearish) et +100 (tres bullish).

DONNEES:
{data_json}

Reponds EXCLUSIVEMENT en JSON:
{{"scores": [{{"symbol": "BTC/USDT", "score": 25, "label": "mildly_bullish"}}, ...]}}

Labels possibles: extreme_bearish, bearish, mildly_bearish, neutral, mildly_bullish, bullish, extreme_bullish"""


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SentimentScore:
    """Score de sentiment pour un actif."""
    symbol: str
    composite_score: float = 0.0    # -100 a +100
    momentum_score: float = 0.0     # -100 a +100
    mean_reversion_score: float = 0.0  # -100 a +100
    volatility_regime: str = "normal"  # low, normal, high, extreme
    volatility_score: float = 0.0   # -100 a +100
    ai_score: float | None = None   # Score IA si disponible
    label: str = "neutral"          # Label textuel


@dataclass
class SentimentReport:
    """Rapport complet de sentiment marche."""
    scores: list[SentimentScore] = field(default_factory=list)
    global_sentiment: float = 0.0   # Sentiment global du marche
    global_label: str = "neutral"
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    ai_used: bool = False
    model_used: str = "rule-based"


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _momentum_score(signal: dict) -> float:
    """Score de momentum base sur les variations de prix.

    Combine variations 1h, 24h, 7d avec poids decroissants.
    Retourne un score entre -100 et +100.
    """
    change_1h = signal.get("change_1h", 0) or 0
    change_24h = signal.get("change_24h", 0) or 0
    change_7d = signal.get("change_7d", 0) or 0

    # Poids: 1h (court terme) = 0.2, 24h (moyen terme) = 0.5, 7d (tendance) = 0.3
    raw = change_1h * 0.2 + change_24h * 0.5 + change_7d * 0.3

    # Normaliser dans [-100, +100] avec saturation a +-20%
    normalized = np.clip(raw / 20 * 100, -100, 100)

    return round(float(normalized), 1)


def _mean_reversion_score(signal: dict) -> float:
    """Score de mean reversion: detecte les conditions de sur-achat/sur-vente.

    Si le prix s'est beaucoup eloigne de sa moyenne (forte variation recente),
    la mean reversion predit un retour, donc le score est oppose au momentum.

    Retourne un score entre -100 et +100.
    """
    change_24h = signal.get("change_24h", 0) or 0
    change_7d = signal.get("change_7d", 0) or 0

    # L'ecart entre variation court terme et long terme indique le potentiel de reversion
    deviation = change_24h - change_7d * 0.3

    # Plus la deviation est grande, plus la reversion est probable
    # Score inverse: deviation positive → score negatif (retour attendu a la baisse)
    reversion = -deviation

    # Normaliser
    normalized = np.clip(reversion / 10 * 100, -100, 100)

    return round(float(normalized), 1)


def _volatility_score(signal: dict) -> tuple[float, str]:
    """Score de volatilite et detection du regime.

    Haute volatilite = incertitude = score neutre/negatif.
    Basse volatilite = stabilite = score leger positif.

    Returns:
        (score, regime)
    """
    vol = abs(signal.get("volatility", 0) or 0)

    if vol < VOL_LOW:
        return 20.0, "low"
    elif vol < VOL_NORMAL:
        return 0.0, "normal"
    elif vol < VOL_HIGH:
        return -30.0, "high"
    else:
        return -70.0, "extreme"


def _compute_label(score: float) -> str:
    """Convertit un score numerique en label textuel."""
    if score > 60:
        return "extreme_bullish"
    elif score > 30:
        return "bullish"
    elif score > 10:
        return "mildly_bullish"
    elif score > -10:
        return "neutral"
    elif score > -30:
        return "mildly_bearish"
    elif score > -60:
        return "bearish"
    else:
        return "extreme_bearish"


def _rule_based_sentiment(signals: list[dict]) -> list[SentimentScore]:
    """Calcul rule-based du sentiment pour chaque actif."""
    scores = []

    for sig in signals:
        momentum = _momentum_score(sig)
        reversion = _mean_reversion_score(sig)
        vol_score, vol_regime = _volatility_score(sig)

        # Score composite pondere (sans IA)
        effective_weight = MOMENTUM_WEIGHT + MEAN_REVERSION_WEIGHT + VOLATILITY_WEIGHT
        composite = (
            momentum * MOMENTUM_WEIGHT +
            reversion * MEAN_REVERSION_WEIGHT +
            vol_score * VOLATILITY_WEIGHT
        ) / effective_weight

        composite = round(np.clip(composite, -100, 100), 1)

        scores.append(SentimentScore(
            symbol=sig.get("symbol", "?"),
            composite_score=float(composite),
            momentum_score=momentum,
            mean_reversion_score=reversion,
            volatility_regime=vol_regime,
            volatility_score=vol_score,
            label=_compute_label(composite),
        ))

    return scores


async def _ai_sentiment_classification(signals: list[dict]) -> dict[str, float] | None:
    """Classification IA via Ollama OL1 (qwen3:1.7b) — rapide et leger.

    Retourne un dict {symbol: score} ou None si indisponible.
    """
    # Preparer les donnees simplifiees pour le prompt
    data_for_ai = [
        {
            "symbol": s.get("symbol", "?"),
            "price": s.get("price", 0),
            "change_24h": s.get("change_24h", 0),
            "volatility": s.get("volatility", 0),
            "direction": s.get("direction", "neutral"),
        }
        for s in signals[:10]  # Limiter a 10 pour la vitesse
    ]

    prompt = OL1_SENTIMENT_PROMPT.format(
        data_json=json.dumps(data_for_ai, indent=2, default=str)
    )

    result = await query_ollama(
        prompt,
        node_name="OL1",
        system="Tu es un analyste de sentiment marche. Reponds UNIQUEMENT en JSON valide.",
    )

    if not result.get("ok"):
        console.print(f"  [yellow]OL1 sentiment indisponible:[/] {result.get('error', '?')}")
        return None

    # Parser la reponse
    content = result.get("content", "")
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(content[start:end])
            ai_scores = {}
            for entry in parsed.get("scores", []):
                symbol = entry.get("symbol", "")
                score = entry.get("score", 0)
                if symbol and isinstance(score, (int, float)):
                    ai_scores[symbol] = float(np.clip(score, -100, 100))

            console.print(f"  [dim]OL1 sentiment: {len(ai_scores)} scores, {result.get('latency_ms', 0)}ms[/]")
            return ai_scores
    except (json.JSONDecodeError, ValueError, KeyError):
        console.print("  [yellow]OL1 reponse non parseable[/]")

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT RUN — Point d'entree principal
# ═══════════════════════════════════════════════════════════════════════════════

async def run(run_id: str, signals: list[dict]) -> StepResult:
    """Execute l'agent Sentiment Scorer: scoring multi-facteur + IA.

    Args:
        run_id: Identifiant du run pipeline
        signals: Liste de MarketSignal dicts (symbol, price, change_1h, change_24h, change_7d, volatility, ...)

    Returns:
        StepResult avec SentimentReport dans data
    """
    t0 = time.monotonic()
    console.print("[bold cyan]Agent Sentinel — Sentiment Scorer[/] analyse en cours...")

    if not signals:
        latency = (time.monotonic() - t0) * 1000
        console.print("[yellow]Sentiment Scorer: aucun signal a analyser[/]")
        return StepResult(
            step_name="sentiment_scoring",
            status=StepStatus.FAILED,
            error="Aucun signal fourni",
            agent_used="sentiment_scorer",
            latency_ms=latency,
        )

    # === Etape 1: Scoring rule-based ===
    scores = _rule_based_sentiment(signals)
    model_used = "rule-based"
    ai_used = False

    # === Etape 2: Classification IA (OL1 — best-effort) ===
    ai_scores = await _ai_sentiment_classification(signals)

    if ai_scores:
        ai_used = True
        model_used = "rule-based+OL1-qwen3:1.7b"

        # Integrer les scores IA dans le composite
        for score_obj in scores:
            ai_val = ai_scores.get(score_obj.symbol)
            if ai_val is not None:
                score_obj.ai_score = ai_val
                # Re-calculer le composite avec le poids IA
                old_composite = score_obj.composite_score
                effective_total = MOMENTUM_WEIGHT + MEAN_REVERSION_WEIGHT + VOLATILITY_WEIGHT + AI_SENTIMENT_WEIGHT
                new_composite = (
                    score_obj.momentum_score * MOMENTUM_WEIGHT +
                    score_obj.mean_reversion_score * MEAN_REVERSION_WEIGHT +
                    score_obj.volatility_score * VOLATILITY_WEIGHT +
                    ai_val * AI_SENTIMENT_WEIGHT
                ) / effective_total

                score_obj.composite_score = round(float(np.clip(new_composite, -100, 100)), 1)
                score_obj.label = _compute_label(score_obj.composite_score)

    # === Etape 3: Agregation globale ===
    if scores:
        global_sentiment = round(sum(s.composite_score for s in scores) / len(scores), 1)
    else:
        global_sentiment = 0.0

    global_label = _compute_label(global_sentiment)
    bullish_count = sum(1 for s in scores if s.composite_score > 10)
    bearish_count = sum(1 for s in scores if s.composite_score < -10)
    neutral_count = len(scores) - bullish_count - bearish_count

    report = SentimentReport(
        scores=scores,
        global_sentiment=global_sentiment,
        global_label=global_label,
        bullish_count=bullish_count,
        bearish_count=bearish_count,
        neutral_count=neutral_count,
        ai_used=ai_used,
        model_used=model_used,
    )

    latency = (time.monotonic() - t0) * 1000

    # === Audit DB ===
    db.save_audit(
        run_id, "sentiment_scoring", "sentiment_scorer",
        input_summary=f"{len(signals)} signaux analyses",
        output_summary=(
            f"sentiment_global={global_sentiment:+.0f} ({global_label}), "
            f"bullish={bullish_count}, bearish={bearish_count}, neutral={neutral_count}"
        ),
        model_used=model_used,
        latency_ms=latency,
    )

    # === Affichage ===
    _print_sentiment_report(report)

    # Confidence: haute si consensus fort (peu de neutral), basse si mixte
    consensus_strength = abs(bullish_count - bearish_count) / max(len(scores), 1) * 100
    confidence = round(min(95, 50 + consensus_strength * 0.5), 1)

    console.print(
        f"[green]Sentiment Scorer done[/] — global={global_sentiment:+.0f} ({global_label}), "
        f"B={bullish_count}/N={neutral_count}/S={bearish_count} in {int(latency)}ms [{model_used}]"
    )

    return StepResult(
        step_name="sentiment_scoring",
        status=StepStatus.SUCCESS,
        data={
            "scores": [
                {
                    "symbol": s.symbol,
                    "composite_score": s.composite_score,
                    "momentum_score": s.momentum_score,
                    "mean_reversion_score": s.mean_reversion_score,
                    "volatility_regime": s.volatility_regime,
                    "volatility_score": s.volatility_score,
                    "ai_score": s.ai_score,
                    "label": s.label,
                }
                for s in scores
            ],
            "global_sentiment": report.global_sentiment,
            "global_label": report.global_label,
            "bullish_count": report.bullish_count,
            "bearish_count": report.bearish_count,
            "neutral_count": report.neutral_count,
            "ai_used": report.ai_used,
            "model_used": report.model_used,
        },
        confidence=confidence,
        agent_used="sentiment_scorer",
        model_used=model_used,
        latency_ms=latency,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _print_sentiment_report(report: SentimentReport) -> None:
    """Affiche le rapport de sentiment."""
    if not report.scores:
        console.print("[yellow]Aucun score de sentiment[/]")
        return

    # Table des scores par actif
    table = Table(title="Sentiment Scores", show_lines=False)
    table.add_column("Symbol", style="cyan")
    table.add_column("Composite", justify="right")
    table.add_column("Momentum", justify="right")
    table.add_column("Reversion", justify="right")
    table.add_column("Vol Regime")
    table.add_column("AI Score", justify="right")
    table.add_column("Label")

    for s in report.scores[:15]:
        # Couleur du score composite
        if s.composite_score > 10:
            color = "green"
        elif s.composite_score < -10:
            color = "red"
        else:
            color = "yellow"

        # Label avec couleur
        label_colors = {
            "extreme_bullish": "bold green",
            "bullish": "green",
            "mildly_bullish": "green",
            "neutral": "yellow",
            "mildly_bearish": "red",
            "bearish": "red",
            "extreme_bearish": "bold red",
        }
        label_color = label_colors.get(s.label, "white")

        ai_str = f"{s.ai_score:+.0f}" if s.ai_score is not None else "[dim]---[/]"

        table.add_row(
            s.symbol,
            f"[{color}]{s.composite_score:+.0f}[/]",
            f"{s.momentum_score:+.0f}",
            f"{s.mean_reversion_score:+.0f}",
            s.volatility_regime,
            ai_str,
            f"[{label_color}]{s.label}[/]",
        )

    console.print(table)

    # Panel de synthese
    sentiment_color = "green" if report.global_sentiment > 10 else ("red" if report.global_sentiment < -10 else "yellow")
    console.print(Panel(
        f"[bold]Sentiment Global:[/] [{sentiment_color}]{report.global_sentiment:+.0f} ({report.global_label})[/]\n"
        f"[bold]Bullish:[/] [green]{report.bullish_count}[/]  "
        f"[bold]Neutral:[/] [yellow]{report.neutral_count}[/]  "
        f"[bold]Bearish:[/] [red]{report.bearish_count}[/]\n"
        f"[bold]IA utilisee:[/] {'OUI (OL1 qwen3:1.7b)' if report.ai_used else 'NON (rule-based)'}\n"
        f"[bold]Modele:[/] {report.model_used}",
        title="[bold]Sentiment Scorer — Synthese[/]",
        border_style=sentiment_color,
    ))

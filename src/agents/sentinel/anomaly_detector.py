"""Agent Sentinel — Anomaly Detector: detection statistique d'anomalies marche.

Analyse les signaux de marche pour identifier:
- Outliers statistiques (z-score > 2.5)
- Spikes de volume (>3x moyenne)
- Divergences prix-volume
- Patterns de contagion cross-asset (correlations)

Utilise numpy pour les calculs statistiques.
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
# CONFIGURATION — Seuils de detection
# ═══════════════════════════════════════════════════════════════════════════════

ZSCORE_THRESHOLD = 2.5          # Seuil z-score pour outlier
VOLUME_SPIKE_MULTIPLIER = 3.0   # Volume > 3x moyenne = spike
DIVERGENCE_THRESHOLD = 0.5      # Seuil de divergence prix-volume
CORRELATION_THRESHOLD = 0.75    # Seuil de correlation pour contagion
MIN_SIGNALS_FOR_STATS = 3       # Minimum de signaux pour analyse statistique


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Anomaly:
    """Une anomalie detectee dans les signaux de marche."""
    anomaly_type: str           # "zscore_outlier", "volume_spike", "divergence", "contagion"
    symbol: str
    severity: str               # "low", "medium", "high", "critical"
    score: float                # Score de l'anomalie (0-100)
    description: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class AnomalyReport:
    """Rapport complet de detection d'anomalies."""
    anomalies: list[Anomaly] = field(default_factory=list)
    total_signals_analyzed: int = 0
    anomaly_rate: float = 0.0   # % de signaux anormaux
    contagion_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    market_stress_index: float = 0.0  # 0-100, indicateur de stress global


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTORS
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_zscore_outliers(signals: list[dict]) -> list[Anomaly]:
    """Detecte les outliers statistiques via z-score sur les variations 24h."""
    anomalies = []
    changes = [s.get("change_24h", 0) for s in signals if s.get("change_24h") is not None]

    if len(changes) < MIN_SIGNALS_FOR_STATS:
        return anomalies

    arr = np.array(changes, dtype=np.float64)
    mean = np.mean(arr)
    std = np.std(arr)

    if std < 1e-10:  # Ecart-type trop faible, pas d'outliers
        return anomalies

    for sig in signals:
        change = sig.get("change_24h", 0)
        if change is None:
            continue

        zscore = abs((change - mean) / std)

        if zscore > ZSCORE_THRESHOLD:
            severity = "critical" if zscore > 4.0 else ("high" if zscore > 3.0 else "medium")
            score = min(100, zscore * 20)
            anomalies.append(Anomaly(
                anomaly_type="zscore_outlier",
                symbol=sig.get("symbol", "?"),
                severity=severity,
                score=round(score, 1),
                description=(
                    f"Variation 24h anormale: {change:+.2f}% "
                    f"(z-score={zscore:.2f}, moyenne={mean:.2f}%, std={std:.2f}%)"
                ),
                metrics={"zscore": round(zscore, 3), "change_24h": change, "mean": round(mean, 3), "std": round(std, 3)},
            ))

    return anomalies


def _detect_volume_spikes(signals: list[dict]) -> list[Anomaly]:
    """Detecte les spikes de volume (>3x la moyenne)."""
    anomalies = []
    volumes = [s.get("volume_24h", 0) for s in signals if s.get("volume_24h", 0) > 0]

    if len(volumes) < MIN_SIGNALS_FOR_STATS:
        return anomalies

    avg_volume = np.mean(volumes)

    if avg_volume < 1e-10:
        return anomalies

    for sig in signals:
        vol = sig.get("volume_24h", 0)
        if vol <= 0:
            continue

        ratio = vol / avg_volume

        if ratio > VOLUME_SPIKE_MULTIPLIER:
            severity = "critical" if ratio > 8 else ("high" if ratio > 5 else "medium")
            score = min(100, ratio * 12)
            anomalies.append(Anomaly(
                anomaly_type="volume_spike",
                symbol=sig.get("symbol", "?"),
                severity=severity,
                score=round(score, 1),
                description=(
                    f"Spike de volume: {ratio:.1f}x la moyenne "
                    f"(volume={vol:,.0f}, moyenne={avg_volume:,.0f})"
                ),
                metrics={"volume": vol, "avg_volume": round(avg_volume, 0), "ratio": round(ratio, 2)},
            ))

    return anomalies


def _detect_price_volume_divergence(signals: list[dict]) -> list[Anomaly]:
    """Detecte les divergences prix-volume (prix monte mais volume baisse, ou inversement)."""
    anomalies = []

    for sig in signals:
        change = sig.get("change_24h", 0)
        vol = sig.get("volume_24h", 0)
        volatility = sig.get("volatility", 0)

        if change is None or vol <= 0:
            continue

        # Divergence: forte variation de prix avec faible volume = mouvement suspect
        # Ou: faible variation de prix avec volume enorme = accumulation/distribution
        price_magnitude = abs(change)
        vol_magnitude = abs(volatility)

        if price_magnitude > 5 and vol_magnitude < 1:
            # Prix bouge beaucoup, volatilite faible — mouvement a faible conviction
            score = min(100, price_magnitude * 8)
            anomalies.append(Anomaly(
                anomaly_type="divergence",
                symbol=sig.get("symbol", "?"),
                severity="high" if score > 60 else "medium",
                score=round(score, 1),
                description=(
                    f"Divergence prix-volume: variation {change:+.2f}% "
                    f"avec volatilite faible ({volatility:.2f}%)"
                ),
                metrics={"change_24h": change, "volatility": volatility, "volume": vol},
            ))

        elif price_magnitude < 1 and vol_magnitude > 5:
            # Prix stable mais volatilite elevee — accumulation/distribution silencieuse
            score = min(100, vol_magnitude * 10)
            anomalies.append(Anomaly(
                anomaly_type="divergence",
                symbol=sig.get("symbol", "?"),
                severity="medium",
                score=round(score, 1),
                description=(
                    f"Accumulation/Distribution: prix stable ({change:+.2f}%) "
                    f"mais volatilite elevee ({volatility:.2f}%)"
                ),
                metrics={"change_24h": change, "volatility": volatility, "volume": vol},
            ))

    return anomalies


def _detect_contagion(signals: list[dict]) -> tuple[list[Anomaly], list[tuple[str, str, float]]]:
    """Detecte les patterns de contagion cross-asset via correlations.

    Si plusieurs actifs de classes differentes bougent dans la meme direction
    avec une amplitude similaire, cela suggere une contagion systemique.
    """
    anomalies = []
    contagion_pairs: list[tuple[str, str, float]] = []

    # Grouper par classe d'actifs
    by_class: dict[str, list[dict]] = {}
    for sig in signals:
        ac = sig.get("asset_class", "unknown")
        if ac not in by_class:
            by_class[ac] = []
        by_class[ac].append(sig)

    # Calculer la correlation entre classes d'actifs (via variations 24h)
    classes = list(by_class.keys())
    if len(classes) < 2:
        return anomalies, contagion_pairs

    class_changes: dict[str, float] = {}
    for ac, sigs in by_class.items():
        changes = [s.get("change_24h", 0) for s in sigs if s.get("change_24h") is not None]
        if changes:
            class_changes[ac] = float(np.mean(changes))

    # Verifier la correlation entre paires de classes
    ac_list = list(class_changes.keys())
    for i in range(len(ac_list)):
        for j in range(i + 1, len(ac_list)):
            ac1, ac2 = ac_list[i], ac_list[j]
            change1, change2 = class_changes[ac1], class_changes[ac2]

            # Si les deux bougent dans la meme direction avec amplitude significative
            if abs(change1) > 1 and abs(change2) > 1:
                # Correlation simplifiee: meme signe et amplitudes proches
                same_direction = (change1 > 0) == (change2 > 0)
                amplitude_ratio = min(abs(change1), abs(change2)) / max(abs(change1), abs(change2))

                if same_direction and amplitude_ratio > DIVERGENCE_THRESHOLD:
                    correlation = amplitude_ratio
                    contagion_pairs.append((ac1, ac2, round(correlation, 3)))

                    if correlation > CORRELATION_THRESHOLD:
                        score = min(100, correlation * 80)
                        anomalies.append(Anomaly(
                            anomaly_type="contagion",
                            symbol=f"{ac1}/{ac2}",
                            severity="high" if correlation > 0.85 else "medium",
                            score=round(score, 1),
                            description=(
                                f"Contagion cross-asset: {ac1} ({change1:+.2f}%) et "
                                f"{ac2} ({change2:+.2f}%) correles a {correlation:.0%}"
                            ),
                            metrics={"correlation": correlation, "change_1": change1, "change_2": change2},
                        ))

    return anomalies, contagion_pairs


def _calculate_stress_index(anomalies: list[Anomaly], total_signals: int) -> float:
    """Calcule un indice de stress global du marche (0-100).

    Base sur la densite et la severite des anomalies detectees.
    """
    if not anomalies or total_signals == 0:
        return 0.0

    severity_weights = {"low": 1, "medium": 2, "high": 4, "critical": 8}
    total_weight = sum(severity_weights.get(a.severity, 1) for a in anomalies)

    # Densite: ratio anomalies / signaux (normalise)
    density = len(anomalies) / max(total_signals, 1)

    # Score de stress: combine densite et severite
    stress = min(100, (density * 50) + (total_weight * 3))

    return round(stress, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT RUN — Point d'entree principal
# ═══════════════════════════════════════════════════════════════════════════════

async def run(run_id: str, signals: list[dict]) -> StepResult:
    """Execute l'agent Anomaly Detector: detection statistique multi-facteur.

    Args:
        run_id: Identifiant du run pipeline
        signals: Liste de MarketSignal dicts (symbol, change_24h, volume_24h, volatility, ...)

    Returns:
        StepResult avec AnomalyReport dans data
    """
    t0 = time.monotonic()
    console.print("[bold cyan]Agent Sentinel — Anomaly Detector[/] analyse en cours...")

    all_anomalies: list[Anomaly] = []

    # === Detection z-score ===
    zscore_anomalies = _detect_zscore_outliers(signals)
    all_anomalies.extend(zscore_anomalies)
    console.print(f"  [dim]Z-score outliers: {len(zscore_anomalies)} detectes[/]")

    # === Detection spikes de volume ===
    volume_anomalies = _detect_volume_spikes(signals)
    all_anomalies.extend(volume_anomalies)
    console.print(f"  [dim]Volume spikes: {len(volume_anomalies)} detectes[/]")

    # === Detection divergences ===
    divergence_anomalies = _detect_price_volume_divergence(signals)
    all_anomalies.extend(divergence_anomalies)
    console.print(f"  [dim]Divergences: {len(divergence_anomalies)} detectees[/]")

    # === Detection contagion ===
    contagion_anomalies, contagion_pairs = _detect_contagion(signals)
    all_anomalies.extend(contagion_anomalies)
    console.print(f"  [dim]Contagion: {len(contagion_anomalies)} patterns, {len(contagion_pairs)} paires[/]")

    # === Calcul indice de stress ===
    stress_index = _calculate_stress_index(all_anomalies, len(signals))

    # Trier par score decroissant
    all_anomalies.sort(key=lambda a: a.score, reverse=True)

    # Construire le rapport
    anomaly_rate = round(len(all_anomalies) / max(len(signals), 1) * 100, 1)

    report = AnomalyReport(
        anomalies=all_anomalies,
        total_signals_analyzed=len(signals),
        anomaly_rate=anomaly_rate,
        contagion_pairs=contagion_pairs,
        market_stress_index=stress_index,
    )

    latency = (time.monotonic() - t0) * 1000

    # === Audit DB ===
    db.save_audit(
        run_id, "anomaly_detection", "anomaly_detector",
        input_summary=f"{len(signals)} signaux analyses",
        output_summary=(
            f"{len(all_anomalies)} anomalies ({anomaly_rate:.1f}%), "
            f"stress={stress_index:.0f}, contagion={len(contagion_pairs)} paires"
        ),
        model_used="numpy-statistical",
        latency_ms=latency,
    )

    # === Affichage ===
    _print_anomaly_report(report)

    status = StepStatus.SUCCESS
    confidence = max(0, 100 - stress_index * 0.6)

    console.print(
        f"[green]Anomaly Detector done[/] — {len(all_anomalies)} anomalies, "
        f"stress={stress_index:.0f}, taux={anomaly_rate:.1f}% in {int(latency)}ms"
    )

    return StepResult(
        step_name="anomaly_detection",
        status=status,
        data={
            "anomalies": [
                {
                    "type": a.anomaly_type,
                    "symbol": a.symbol,
                    "severity": a.severity,
                    "score": a.score,
                    "description": a.description,
                    "metrics": a.metrics,
                }
                for a in all_anomalies
            ],
            "total_signals": report.total_signals_analyzed,
            "anomaly_rate": report.anomaly_rate,
            "contagion_pairs": [
                {"asset_1": p[0], "asset_2": p[1], "correlation": p[2]}
                for p in contagion_pairs
            ],
            "market_stress_index": report.market_stress_index,
        },
        confidence=round(confidence, 1),
        agent_used="anomaly_detector",
        model_used="numpy-statistical",
        latency_ms=latency,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _print_anomaly_report(report: AnomalyReport) -> None:
    """Affiche le rapport d'anomalies."""
    if not report.anomalies:
        console.print("[green]Aucune anomalie detectee[/]")
        return

    # Table des anomalies
    table = Table(title=f"Anomalies Detectees ({len(report.anomalies)})", show_lines=False)
    table.add_column("Type", style="cyan")
    table.add_column("Symbol")
    table.add_column("Severity")
    table.add_column("Score", justify="right")
    table.add_column("Description", max_width=60)

    severity_colors = {"low": "dim", "medium": "yellow", "high": "red", "critical": "bold red"}

    for a in report.anomalies[:15]:
        color = severity_colors.get(a.severity, "white")
        table.add_row(
            a.anomaly_type,
            a.symbol,
            f"[{color}]{a.severity.upper()}[/]",
            f"[{color}]{a.score:.0f}[/]",
            a.description[:60],
        )

    console.print(table)

    # Stress index
    stress_color = (
        "red" if report.market_stress_index > 60
        else ("yellow" if report.market_stress_index > 30 else "green")
    )

    console.print(Panel(
        f"[bold]Signaux analyses:[/] {report.total_signals_analyzed}\n"
        f"[bold]Anomalies:[/] {len(report.anomalies)} ({report.anomaly_rate:.1f}%)\n"
        f"[bold]Stress Index:[/] [{stress_color}]{report.market_stress_index:.0f}/100[/]\n"
        f"[bold]Paires contagion:[/] {len(report.contagion_pairs)}",
        title="[bold]Anomaly Report — Synthese[/]",
        border_style=stress_color,
    ))

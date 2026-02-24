"""Agent Sentinel — Alert Agent: generation et dispatch d'alertes de risque.

Evalue les conditions de marche, le profil de risque et les anomalies detectees
pour generer des alertes classifiees par severite:
- INFO: conditions normales avec changement notable
- WARNING: seuils de risque approches (overall_risk > 70)
- CRITICAL: seuils depasses (overall_risk > 85), anomalies critiques
- EMERGENCY: cascade detectee, intervention immediate requise

Simule l'envoi via canaux multiples: webhook, email, Slack-style.
Toutes les alertes sont journalisees en base pour audit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.pipeline_engine import StepResult, StepStatus
from src.config import config
from src import database as db

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Seuils d'alerte
# ═══════════════════════════════════════════════════════════════════════════════

ALERT_RISK_HIGH = 70.0          # overall_risk > 70 = WARNING
ALERT_RISK_CRITICAL = 85.0      # overall_risk > 85 = CRITICAL
ASSET_CLASS_CRITICAL = 80.0     # Risque par classe d'actifs > 80 = CRITICAL
ANOMALY_CRITICAL_SCORE = 80.0   # Score d'anomalie > 80 = CRITICAL alert
MAX_ALERTS_PER_RUN = 50         # Limite pour eviter le spam

# Canaux de dispatch simules
DISPATCH_CHANNELS = ["webhook", "email", "slack"]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Alert:
    """Une alerte generee par l'agent."""
    level: str                  # INFO, WARNING, CRITICAL, EMERGENCY
    category: str               # risk, anomaly, cascade, position, contagion
    title: str
    description: str
    recommended_action: str = ""
    source_agent: str = ""
    symbol: str = ""
    score: float = 0.0
    timestamp: str = ""
    dispatched_to: list[str] = field(default_factory=list)


@dataclass
class AlertReport:
    """Rapport d'alertes complet."""
    alerts: list[Alert] = field(default_factory=list)
    emergency_count: int = 0
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    total_dispatched: int = 0
    channels_used: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# ALERT GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_risk_alerts(risk_profile: dict) -> list[Alert]:
    """Genere des alertes basees sur le profil de risque global."""
    alerts: list[Alert] = []
    overall_risk = risk_profile.get("overall_risk", 0)
    cascade = risk_profile.get("cascade_detected", False)
    cascade_desc = risk_profile.get("cascade_description", "")
    alert_level_str = risk_profile.get("alert_level", "LOW")

    now = datetime.now(timezone.utc).isoformat()

    # === Alerte risque global ===
    if overall_risk >= ALERT_RISK_CRITICAL:
        level = "EMERGENCY" if cascade else "CRITICAL"
        alerts.append(Alert(
            level=level,
            category="risk",
            title=f"Risque global critique: {overall_risk:.0f}/100",
            description=(
                f"Le risque global du portefeuille a atteint {overall_risk:.0f}/100, "
                f"depassant le seuil critique de {ALERT_RISK_CRITICAL:.0f}. "
                f"{'CASCADE ACTIVE: ' + cascade_desc if cascade else 'Revue immediate recommandee.'}"
            ),
            recommended_action=(
                "INTERVENTION IMMEDIATE: Activer les couvertures d'urgence et "
                "convoquer le comite de risque."
                if level == "EMERGENCY" else
                "Reviser toutes les positions ouvertes et activer "
                "les strategies de couverture validees."
            ),
            source_agent="risk_aggregator",
            score=overall_risk,
            timestamp=now,
        ))
    elif overall_risk >= ALERT_RISK_HIGH:
        alerts.append(Alert(
            level="WARNING",
            category="risk",
            title=f"Risque global eleve: {overall_risk:.0f}/100",
            description=(
                f"Le risque global approche du seuil critique "
                f"({ALERT_RISK_CRITICAL:.0f}). Surveillance renforcee requise."
            ),
            recommended_action=(
                "Augmenter la frequence de monitoring. "
                "Preparer les strategies de couverture en standby."
            ),
            source_agent="risk_aggregator",
            score=overall_risk,
            timestamp=now,
        ))

    # === Alertes par classe d'actifs ===
    for asset_class, risk_score in risk_profile.get("risk_by_asset_class", {}).items():
        if risk_score >= ASSET_CLASS_CRITICAL:
            alerts.append(Alert(
                level="CRITICAL",
                category="risk",
                title=f"Classe {asset_class}: risque critique ({risk_score:.0f}/100)",
                description=(
                    f"La classe d'actifs '{asset_class}' a atteint un niveau "
                    f"de risque de {risk_score:.0f}/100."
                ),
                recommended_action=(
                    f"Reduire l'exposition sur {asset_class} ou activer "
                    f"la couverture sectorielle."
                ),
                source_agent="risk_aggregator",
                symbol=asset_class,
                score=risk_score,
                timestamp=now,
            ))
        elif risk_score >= ALERT_RISK_HIGH:
            alerts.append(Alert(
                level="WARNING",
                category="risk",
                title=f"Classe {asset_class}: risque eleve ({risk_score:.0f}/100)",
                description=(
                    f"La classe d'actifs '{asset_class}' montre un risque "
                    f"eleve de {risk_score:.0f}/100."
                ),
                recommended_action=f"Surveiller {asset_class} de pres.",
                source_agent="risk_aggregator",
                symbol=asset_class,
                score=risk_score,
                timestamp=now,
            ))

    # === Alerte cascade ===
    if cascade:
        alerts.append(Alert(
            level="EMERGENCY",
            category="cascade",
            title="CASCADE DE RISQUE DETECTEE",
            description=(
                f"Signaux bearish alignes sur multiples classes d'actifs. "
                f"{cascade_desc}"
            ),
            recommended_action=(
                "PRIORITE MAXIMALE: Evaluer l'exposition totale du portefeuille. "
                "Activer les hedges systemiques. Notifier la direction."
            ),
            source_agent="risk_aggregator",
            score=overall_risk,
            timestamp=now,
        ))

    return alerts


def _generate_anomaly_alerts(anomalies: list[dict]) -> list[Alert]:
    """Genere des alertes basees sur les anomalies detectees."""
    alerts: list[Alert] = []
    now = datetime.now(timezone.utc).isoformat()

    for anomaly in anomalies:
        severity = anomaly.get("severity", "low")
        score = anomaly.get("score", 0)
        atype = anomaly.get("type", "unknown")
        symbol = anomaly.get("symbol", "?")
        description = anomaly.get("description", "Anomalie detectee")

        if severity == "critical" or score >= ANOMALY_CRITICAL_SCORE:
            alerts.append(Alert(
                level="CRITICAL",
                category="anomaly",
                title=f"Anomalie critique: {atype} sur {symbol}",
                description=description,
                recommended_action=(
                    f"Verifier immediatement la position sur {symbol}. "
                    f"Investiguer la source de l'anomalie ({atype})."
                ),
                source_agent="anomaly_detector",
                symbol=symbol,
                score=score,
                timestamp=now,
            ))
        elif severity == "high":
            alerts.append(Alert(
                level="WARNING",
                category="anomaly",
                title=f"Anomalie elevee: {atype} sur {symbol}",
                description=description,
                recommended_action=f"Monitorer {symbol} avec attention.",
                source_agent="anomaly_detector",
                symbol=symbol,
                score=score,
                timestamp=now,
            ))

    # === Alerte contagion (si des anomalies de type contagion existent) ===
    contagion_anomalies = [a for a in anomalies if a.get("type") == "contagion"]
    if len(contagion_anomalies) >= 2:
        alerts.append(Alert(
            level="WARNING",
            category="contagion",
            title=f"Contagion multi-actifs detectee ({len(contagion_anomalies)} paires)",
            description=(
                f"{len(contagion_anomalies)} patterns de contagion cross-asset detectes. "
                f"Risque de propagation systemique."
            ),
            recommended_action=(
                "Reviser les correlations du portefeuille. "
                "Considerer la diversification ou le hedging sectoriel."
            ),
            source_agent="anomaly_detector",
            score=max((a.get("score", 0) for a in contagion_anomalies), default=0),
            timestamp=now,
        ))

    return alerts


# ═══════════════════════════════════════════════════════════════════════════════
# DISPATCH — Simulation d'envoi multi-canal
# ═══════════════════════════════════════════════════════════════════════════════

def _dispatch_alerts(alerts: list[Alert]) -> tuple[int, list[str]]:
    """Simule l'envoi des alertes via les canaux configures.

    En production, ceci serait connecte a de vrais webhooks, SMTP, etc.

    Returns:
        (nombre total dispatche, liste des canaux utilises)
    """
    channels_used: set[str] = set()
    total_dispatched = 0

    for alert in alerts:
        target_channels = _get_dispatch_channels(alert.level)
        alert.dispatched_to = target_channels
        channels_used.update(target_channels)

        for channel in target_channels:
            _simulate_dispatch(channel, alert)
            total_dispatched += 1

    return total_dispatched, sorted(channels_used)


def _get_dispatch_channels(level: str) -> list[str]:
    """Determine les canaux de dispatch selon le niveau d'alerte."""
    if level == "EMERGENCY":
        return ["webhook", "email", "slack"]  # Tous les canaux
    if level == "CRITICAL":
        return ["webhook", "slack"]
    if level == "WARNING":
        return ["slack"]
    return []  # INFO: log uniquement


def _simulate_dispatch(channel: str, alert: Alert) -> None:
    """Simule l'envoi d'une alerte sur un canal.

    En production, ceci appellerait les APIs reelles.
    """
    # Log pour demo/audit
    console.print(
        f"    [dim]-> {channel.upper()}: [{alert.level}] {alert.title[:50]}[/]"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT RUN — Point d'entree principal
# ═══════════════════════════════════════════════════════════════════════════════

async def run(
    run_id: str,
    risk_profile: dict,
    anomalies: list[dict] | None = None,
) -> StepResult:
    """Execute l'Alert Agent: evaluation des seuils et generation d'alertes.

    Args:
        run_id: Identifiant du run pipeline
        risk_profile: Profil de risque du risk_aggregator
            (overall_risk, risk_by_asset_class, cascade_detected, ...)
        anomalies: Liste d'anomalies du anomaly_detector (optionnel)

    Returns:
        StepResult avec AlertReport dans data
    """
    t0 = time.monotonic()
    console.print("[bold cyan]Agent Sentinel — Alert Agent[/] evaluation en cours...")

    all_alerts: list[Alert] = []

    # === Etape 1: Alertes de risque ===
    risk_alerts = _generate_risk_alerts(risk_profile)
    all_alerts.extend(risk_alerts)
    console.print(f"  [dim]Alertes risque: {len(risk_alerts)}[/]")

    # === Etape 2: Alertes d'anomalies ===
    if anomalies:
        anomaly_alerts = _generate_anomaly_alerts(anomalies)
        all_alerts.extend(anomaly_alerts)
        console.print(f"  [dim]Alertes anomalies: {len(anomaly_alerts)}[/]")

    # === Limiter le nombre d'alertes ===
    if len(all_alerts) > MAX_ALERTS_PER_RUN:
        console.print(
            f"  [yellow]Trop d'alertes ({len(all_alerts)}), "
            f"limite a {MAX_ALERTS_PER_RUN}[/]"
        )
        # Garder les plus critiques en priorite
        all_alerts.sort(key=lambda a: _level_priority(a.level))
        all_alerts = all_alerts[:MAX_ALERTS_PER_RUN]

    # === Etape 3: Trier par severite ===
    all_alerts.sort(key=lambda a: _level_priority(a.level))

    # === Etape 4: Dispatch (simulation) ===
    total_dispatched = 0
    channels_used: list[str] = []
    if all_alerts:
        console.print(f"  [dim]Dispatch des alertes...[/]")
        total_dispatched, channels_used = _dispatch_alerts(all_alerts)

    # === Compter par niveau ===
    emergency_count = sum(1 for a in all_alerts if a.level == "EMERGENCY")
    critical_count = sum(1 for a in all_alerts if a.level == "CRITICAL")
    warning_count = sum(1 for a in all_alerts if a.level == "WARNING")
    info_count = sum(1 for a in all_alerts if a.level == "INFO")

    report = AlertReport(
        alerts=all_alerts,
        emergency_count=emergency_count,
        critical_count=critical_count,
        warning_count=warning_count,
        info_count=info_count,
        total_dispatched=total_dispatched,
        channels_used=channels_used,
    )

    latency = (time.monotonic() - t0) * 1000

    # === Audit DB ===
    db.save_audit(
        run_id, "alert_generation", "alert_agent",
        input_summary=(
            f"risk_profile (risk={risk_profile.get('overall_risk', 0):.0f}) "
            f"+ {len(anomalies or [])} anomalies"
        ),
        output_summary=(
            f"{len(all_alerts)} alertes "
            f"(emergency={emergency_count}, critical={critical_count}, "
            f"warning={warning_count}), "
            f"dispatched={total_dispatched} via {','.join(channels_used) or 'none'}"
        ),
        model_used="rule-based",
        latency_ms=latency,
    )

    # === Affichage ===
    _print_alert_report(report)

    # Confidence: haute si peu d'alertes critiques (systeme stable)
    confidence = 90.0
    if emergency_count > 0:
        confidence = 60.0
    elif critical_count > 0:
        confidence = 70.0
    elif warning_count > 0:
        confidence = 80.0

    console.print(
        f"[green]Alert Agent done[/] — {len(all_alerts)} alertes "
        f"(E={emergency_count}/C={critical_count}/W={warning_count}/I={info_count}), "
        f"dispatched={total_dispatched} in {int(latency)}ms"
    )

    return StepResult(
        step_name="alert_generation",
        status=StepStatus.SUCCESS,
        data={
            "alerts": [
                {
                    "level": a.level,
                    "category": a.category,
                    "title": a.title,
                    "description": a.description,
                    "recommended_action": a.recommended_action,
                    "source_agent": a.source_agent,
                    "symbol": a.symbol,
                    "score": a.score,
                    "timestamp": a.timestamp,
                    "dispatched_to": a.dispatched_to,
                }
                for a in all_alerts
            ],
            "emergency_count": emergency_count,
            "critical_count": critical_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "total_alerts": len(all_alerts),
            "total_dispatched": total_dispatched,
            "channels_used": channels_used,
        },
        confidence=confidence,
        agent_used="alert_agent",
        model_used="rule-based",
        latency_ms=latency,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _level_priority(level: str) -> int:
    """Retourne la priorite numerique d'un niveau d'alerte (0 = plus urgent)."""
    return {"EMERGENCY": 0, "CRITICAL": 1, "WARNING": 2, "INFO": 3}.get(level, 99)


# ═══════════════════════════════════════════════════════════════════════════════
# AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _print_alert_report(report: AlertReport) -> None:
    """Affiche le rapport d'alertes complet."""
    if not report.alerts:
        console.print("[green]Aucune alerte generee — conditions normales[/]")
        return

    # Table des alertes
    table = Table(
        title=f"Alertes Generees ({len(report.alerts)})",
        show_lines=False,
    )
    table.add_column("Level", width=12)
    table.add_column("Category", style="cyan", width=12)
    table.add_column("Title", max_width=45)
    table.add_column("Score", justify="right", width=6)
    table.add_column("Dispatch", style="dim", max_width=20)

    level_colors = {
        "EMERGENCY": "bold red",
        "CRITICAL": "red",
        "WARNING": "yellow",
        "INFO": "dim",
    }

    for a in report.alerts[:20]:
        color = level_colors.get(a.level, "white")
        dispatch_str = ", ".join(a.dispatched_to) if a.dispatched_to else "[dim]log only[/]"
        table.add_row(
            f"[{color}]{a.level}[/]",
            a.category,
            a.title[:45],
            f"{a.score:.0f}" if a.score else "",
            dispatch_str,
        )

    console.print(table)

    # Details des alertes EMERGENCY/CRITICAL
    critical_alerts = [a for a in report.alerts if a.level in ("EMERGENCY", "CRITICAL")]
    if critical_alerts:
        for a in critical_alerts[:5]:
            color = level_colors.get(a.level, "white")
            console.print(Panel(
                f"[bold]Categorie:[/] {a.category}\n"
                f"[bold]Description:[/] {a.description}\n"
                f"[bold]Action recommandee:[/] {a.recommended_action}\n"
                f"[bold]Source:[/] {a.source_agent}\n"
                f"[bold]Score:[/] {a.score:.0f}",
                title=f"[{color}]{a.level}: {a.title}[/]",
                border_style=color.replace("bold ", ""),
            ))

    # Panel de synthese
    has_emergency = report.emergency_count > 0
    has_critical = report.critical_count > 0
    if has_emergency:
        border_color = "red"
    elif has_critical:
        border_color = "red"
    elif report.warning_count > 0:
        border_color = "yellow"
    else:
        border_color = "green"

    console.print(Panel(
        f"[bold]Total alertes:[/] {len(report.alerts)}\n"
        f"[bold red]Emergency:[/] {report.emergency_count}\n"
        f"[bold red]Critical:[/] {report.critical_count}\n"
        f"[bold yellow]Warning:[/] {report.warning_count}\n"
        f"[bold dim]Info:[/] {report.info_count}\n"
        f"\n[bold]Dispatched:[/] {report.total_dispatched} messages\n"
        f"[bold]Canaux:[/] {', '.join(report.channels_used) if report.channels_used else 'Aucun'}",
        title="[bold]Alert Agent — Synthese[/]",
        border_style=border_color,
    ))

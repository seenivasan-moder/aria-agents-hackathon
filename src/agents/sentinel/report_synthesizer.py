"""Agent Sentinel — Report Synthesizer: generation de rapport executif final.

Synthetise les resultats de TOUS les agents du pipeline en un rapport
executif structure avec 6 sections:
1. Resume Executif
2. Etat des Marches
3. Exposition Corporate
4. Strategie Recommandee
5. Allocation des Positions
6. Audit & Tracabilite

Mode hybride: Airia pour narrative enrichie (best-effort) + rule-based fallback.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.pipeline_engine import StepResult, StepStatus
from src.config import config
from src.airia_bridge import bridge
from src import database as db

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

REPORT_VERSION = "2.0"
REPORT_TITLE = "Sentinel Risk Intelligence Report"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ReportSection:
    """Une section du rapport executif."""
    number: int
    title: str
    content: str


@dataclass
class ExecutiveReport:
    """Rapport executif complet genere par la synthese."""
    title: str = REPORT_TITLE
    version: str = REPORT_VERSION
    timestamp: str = ""
    run_id: str = ""
    sections: list[ReportSection] = field(default_factory=list)
    full_text: str = ""
    airia_narrative: str = ""
    models_used: list[str] = field(default_factory=list)
    total_agents_used: int = 0
    total_latency_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION BUILDERS — Rule-based synthesis
# ═══════════════════════════════════════════════════════════════════════════════

def _build_executive_summary(all_results: dict) -> ReportSection:
    """Section 1: Resume Executif — 3-5 phrases cles."""
    risk_profile = all_results.get("risk_profile", {})
    sentiment = all_results.get("sentiment", {})
    backtests = all_results.get("backtests", {})
    anomalies = all_results.get("anomalies", {})

    overall_risk = risk_profile.get("overall_risk", 0)
    alert_level = risk_profile.get("alert_level", "UNKNOWN")
    cascade = risk_profile.get("cascade_detected", False)
    global_sentiment = sentiment.get("global_sentiment", 0)
    global_label = sentiment.get("global_label", "neutral")
    anomaly_count = len(anomalies.get("anomalies", []))
    stress_index = anomalies.get("market_stress_index", 0)

    best_strategy = backtests.get("best_strategy", "N/A")
    best_sharpe = backtests.get("best_sharpe", 0)

    lines = []
    lines.append(
        f"Le profil de risque global est evalue a {overall_risk:.0f}/100 "
        f"avec un niveau d'alerte {alert_level}."
    )

    if cascade:
        lines.append(
            f"ALERTE CASCADE: Des signaux de risque correles ont ete detectes "
            f"sur plusieurs classes d'actifs simultanement. "
            f"Description: {risk_profile.get('cascade_description', 'N/A')}."
        )

    lines.append(
        f"Le sentiment global du marche est {global_label} ({global_sentiment:+.0f}), "
        f"avec {anomaly_count} anomalies detectees (stress index: {stress_index:.0f}/100)."
    )

    lines.append(
        f"La strategie recommandee par le backtest est '{best_strategy}' "
        f"avec un Sharpe Ratio de {best_sharpe:.2f}."
    )

    return ReportSection(
        number=1,
        title="RESUME EXECUTIF",
        content="\n".join(lines),
    )


def _build_market_state(all_results: dict) -> ReportSection:
    """Section 2: Etat des Marches — signaux cles et anomalies."""
    signals = all_results.get("signals", [])
    anomalies = all_results.get("anomalies", {})
    sentiment = all_results.get("sentiment", {})

    lines = []

    # Signaux de marche
    if isinstance(signals, list):
        lines.append(f"Nombre de signaux analyses: {len(signals)}")
        # Top 5 par risque
        sorted_sigs = sorted(signals, key=lambda s: s.get("risk_score", 0), reverse=True)
        if sorted_sigs:
            lines.append("\nTop signaux a risque:")
            for sig in sorted_sigs[:5]:
                symbol = sig.get("symbol", "?")
                risk = sig.get("risk_score", 0)
                change = sig.get("change_24h", 0)
                direction = sig.get("direction", "neutral")
                lines.append(
                    f"  - {symbol}: risque {risk:.0f}, "
                    f"variation 24h {change:+.2f}%, direction {direction}"
                )

    # Anomalies
    anomaly_list = anomalies.get("anomalies", [])
    if anomaly_list:
        lines.append(f"\nAnomalies detectees: {len(anomaly_list)}")
        for a in anomaly_list[:5]:
            severity = a.get("severity", "?")
            symbol = a.get("symbol", "?")
            atype = a.get("type", "?")
            desc = a.get("description", "")[:80]
            lines.append(f"  - [{severity.upper()}] {symbol}: {atype} — {desc}")

    # Contagion
    contagion = anomalies.get("contagion_pairs", [])
    if contagion:
        lines.append(f"\nPaires de contagion: {len(contagion)}")
        for p in contagion[:3]:
            lines.append(
                f"  - {p.get('asset_1', '?')} <-> {p.get('asset_2', '?')} "
                f"(corr: {p.get('correlation', 0):.2f})"
            )

    # Sentiment par actif
    scores = sentiment.get("scores", [])
    if scores:
        bullish = sum(1 for s in scores if s.get("composite_score", 0) > 10)
        bearish = sum(1 for s in scores if s.get("composite_score", 0) < -10)
        neutral = len(scores) - bullish - bearish
        lines.append(f"\nSentiment: {bullish} bullish, {neutral} neutral, {bearish} bearish")

    return ReportSection(
        number=2,
        title="ETAT DES MARCHES",
        content="\n".join(lines) if lines else "Aucun signal de marche disponible.",
    )


def _build_corporate_exposure(all_results: dict) -> ReportSection:
    """Section 3: Exposition Corporate — risques par devise/actif."""
    exposure = all_results.get("exposure", {})

    lines = []

    if isinstance(exposure, dict) and exposure:
        company = exposure.get("company_id", config.company_id)
        lines.append(f"Entreprise: {company}")
        lines.append(f"Total Actifs: {exposure.get('total_assets', 0):,.0f} {config.base_currency}")
        lines.append(f"Cash Net: {exposure.get('net_cash', 0):,.0f} {config.base_currency}")
        lines.append(f"Score de Risque: {exposure.get('risk_score', 0):.0f}/100")

        positions = exposure.get("positions", [])
        if positions:
            lines.append("\nPositions par devise:")
            for p in positions:
                if isinstance(p, dict):
                    currency = p.get("currency", "?")
                    net = p.get("net_exposure", 0)
                    pct = p.get("exposure_pct", 0)
                    lines.append(
                        f"  - {currency}: {net:+,.0f} ({pct:.1f}% du portefeuille)"
                    )

        risks = exposure.get("concentration_risks", [])
        if risks:
            lines.append("\nRisques de concentration:")
            for r in risks:
                lines.append(f"  - {r}")
    else:
        lines.append("Aucune donnee d'exposition corporate disponible.")

    return ReportSection(
        number=3,
        title="EXPOSITION CORPORATE",
        content="\n".join(lines),
    )


def _build_strategy_recommendation(all_results: dict) -> ReportSection:
    """Section 4: Strategie Recommandee — consensus + backtest."""
    consensus = all_results.get("consensus", {})
    backtests = all_results.get("backtests", {})

    lines = []

    # Strategies du consensus
    strategies = consensus.get("strategies", [])
    if strategies:
        lines.append(f"Strategies generees par consensus ({len(strategies)}):")
        for s in strategies:
            if isinstance(s, dict):
                name = s.get("name", "?")
                conf = s.get("confidence", 0)
                cost = s.get("cost_estimate_pct", 0)
                reduction = s.get("risk_reduction_pct", 0)
                lines.append(
                    f"  - {name}: confiance {conf:.0f}%, cout {cost:.2f}%, "
                    f"reduction risque {reduction:.0f}%"
                )
                rationale = s.get("rationale", "")
                if rationale:
                    lines.append(f"    Rationale: {rationale[:100]}")

    # Resultats de backtest
    backtest_list = backtests.get("backtests", [])
    if backtest_list:
        sim_days = backtests.get("simulation_days", 30)
        num_sims = backtests.get("num_simulations", 50)
        lines.append(f"\nResultats de backtesting ({sim_days}j, {num_sims} simulations):")
        for bt in backtest_list:
            if isinstance(bt, dict):
                name = bt.get("strategy", "?")
                sharpe = bt.get("sharpe_ratio", 0)
                grade = bt.get("grade", "?")
                pnl = bt.get("pnl_30d_pct", 0)
                max_dd = bt.get("max_drawdown_pct", 0)
                lines.append(
                    f"  - {name}: Sharpe={sharpe:.2f}, P&L={pnl:+.1f}%, "
                    f"MaxDD={max_dd:.1f}%, Grade={grade}"
                )

    # Recommandation finale
    best = backtests.get("best_strategy", "")
    if best:
        lines.append(
            f"\nRECOMMANDATION: Adopter la strategie '{best}' basee sur "
            f"le meilleur Sharpe Ratio du backtest Monte Carlo."
        )

    if not lines:
        lines.append("Aucune strategie disponible.")

    return ReportSection(
        number=4,
        title="STRATEGIE RECOMMANDEE",
        content="\n".join(lines),
    )


def _build_position_allocation(all_results: dict) -> ReportSection:
    """Section 5: Allocation des Positions — tailles recommandees."""
    positions = all_results.get("positions", {})

    lines = []

    if isinstance(positions, dict) and positions:
        balance = positions.get("account_balance", 0)
        total_alloc = positions.get("total_allocated_pct", 0)
        total_risk = positions.get("total_risk_pct", 0)
        corr_adj = positions.get("correlation_adjustment", 1.0)

        lines.append(f"Balance du compte: ${balance:,.0f}")
        lines.append(f"Allocation totale: {total_alloc:.1f}%")
        lines.append(f"Risque total: {total_risk:.1f}%")
        lines.append(f"Ajustement correlation: {corr_adj:.2f}x")

        pos_list = positions.get("positions", [])
        if pos_list:
            lines.append("\nPositions par classe d'actifs:")
            for p in pos_list:
                if isinstance(p, dict):
                    symbol = p.get("symbol", "?")
                    weight = p.get("final_weight", 0)
                    usd = p.get("position_usd", 0)
                    max_loss = p.get("max_loss_usd", 0)
                    leverage = p.get("leverage", 1)
                    method = p.get("method", "kelly+risk_parity")
                    lines.append(
                        f"  - {symbol}: {weight:.2%} (${usd:,.0f}), "
                        f"max loss ${max_loss:,.0f}, leverage {leverage}x [{method}]"
                    )
    else:
        lines.append("Aucune allocation de position disponible.")

    return ReportSection(
        number=5,
        title="ALLOCATION DES POSITIONS",
        content="\n".join(lines),
    )


def _build_audit_summary(all_results: dict, run_id: str, latency: float) -> ReportSection:
    """Section 6: Audit & Tracabilite — agents utilises, sources, latences."""
    lines = []

    lines.append(f"Run ID: {run_id}")
    lines.append(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Latence totale du rapport: {latency:.0f}ms")

    # Compter les agents et modeles
    all_models: set[str] = set()
    agent_count = 0

    for key, value in all_results.items():
        if isinstance(value, dict):
            models = value.get("models_used", [])
            if isinstance(models, list):
                all_models.update(models)
            model = value.get("model_used", "")
            if model:
                all_models.add(model)
            agent_count += 1

    lines.append(f"\nAgents executes: {agent_count}")
    lines.append(
        f"Modeles utilises: {', '.join(sorted(all_models)) if all_models else 'N/A'}"
    )

    # Airia status
    backtests = all_results.get("backtests", {})
    airia_used = backtests.get("airia_enriched", False)
    lines.append(f"Airia SDK: {'Actif' if airia_used else 'Inactif (fallback local)'}")

    # Integrite
    lines.append("\nPiste d'audit:")
    lines.append("  - Tous les resultats intermediaires sont stockes en base SQLite")
    lines.append("  - Chaque etape du pipeline est journalisee avec son latency et modele")
    lines.append(f"  - Base de donnees: {config.db_path}")

    lines.append(
        "\nCERTIFICATION: Ce rapport a ete genere automatiquement par le pipeline "
        "Sentinel et peut etre verifie via le run_id dans la base d'audit."
    )

    return ReportSection(
        number=6,
        title="AUDIT & TRACABILITE",
        content="\n".join(lines),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AIRIA NARRATIVE
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_airia_narrative(all_results: dict) -> str | None:
    """Genere une narrative enrichie via Airia (best-effort).

    Envoie un resume des resultats a Airia pour obtenir un rapport
    redige en langage professionnel style Big Four.

    Returns:
        Texte du rapport Airia ou None si indisponible
    """
    if not bridge.is_available:
        return None

    try:
        # Preparer un resume compact pour Airia
        summary = {
            "risk_level": all_results.get("risk_profile", {}).get("alert_level", "UNKNOWN"),
            "overall_risk": all_results.get("risk_profile", {}).get("overall_risk", 0),
            "cascade": all_results.get("risk_profile", {}).get("cascade_detected", False),
            "sentiment": all_results.get("sentiment", {}).get("global_label", "neutral"),
            "anomaly_count": len(all_results.get("anomalies", {}).get("anomalies", [])),
            "best_strategy": all_results.get("backtests", {}).get("best_strategy", "N/A"),
            "best_sharpe": all_results.get("backtests", {}).get("best_sharpe", 0),
        }

        result = bridge.execute_report_synthesis(summary)

        if result.get("ok"):
            content = result.get("result", "")
            console.print(
                f"  [dim]Airia narrative: {result.get('latency_ms', 0)}ms, "
                f"{len(content)} chars[/]"
            )
            return content

        console.print(f"  [yellow]Airia narrative failed: {result.get('error', '?')}[/]")
    except Exception as e:
        console.print(f"  [yellow]Airia narrative error: {e}[/]")

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def _assemble_report(
    sections: list[ReportSection],
    run_id: str,
    airia_narrative: str | None,
) -> str:
    """Assemble toutes les sections en un rapport texte complet."""
    lines = []

    # Header
    lines.append("=" * 72)
    lines.append(f"  {REPORT_TITLE}")
    lines.append(
        f"  Version {REPORT_VERSION} — "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    lines.append(f"  Run ID: {run_id}")
    lines.append("=" * 72)

    # Sections
    for section in sections:
        lines.append("")
        lines.append(f"  [{section.number}] {section.title}")
        lines.append("-" * 72)
        lines.append(section.content)

    # Airia narrative (si disponible)
    if airia_narrative:
        lines.append("")
        lines.append("  [ANNEXE] ANALYSE NARRATIVE AIRIA")
        lines.append("-" * 72)
        lines.append(airia_narrative)

    lines.append("")
    lines.append("=" * 72)
    lines.append("  FIN DU RAPPORT")
    lines.append("=" * 72)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT RUN — Point d'entree principal
# ═══════════════════════════════════════════════════════════════════════════════

async def run(
    run_id: str,
    all_results: dict,
) -> StepResult:
    """Execute l'agent Report Synthesizer: synthese executive multi-agent.

    Args:
        run_id: Identifiant du run pipeline
        all_results: Dict avec les resultats de tous les agents:
            - signals: list[dict] (market signals)
            - exposure: dict (corporate exposure)
            - anomalies: dict (anomaly report)
            - sentiment: dict (sentiment report)
            - risk_profile: dict (risk aggregation)
            - consensus: dict (strategies)
            - positions: dict (position sizing)
            - backtests: dict (backtest results)

    Returns:
        StepResult avec le rapport complet dans data
    """
    t0 = time.monotonic()
    console.print("[bold cyan]Agent Sentinel — Report Synthesizer[/] generation en cours...")

    if not all_results:
        latency = (time.monotonic() - t0) * 1000
        console.print("[yellow]Report Synthesizer: aucun resultat a synthetiser[/]")
        return StepResult(
            step_name="report_synthesis",
            status=StepStatus.FAILED,
            error="Aucun resultat fourni",
            agent_used="report_synthesizer",
            latency_ms=latency,
        )

    models_used = ["rule-based"]

    # === Etape 1: Construire les 6 sections ===
    console.print("  [dim]Construction des sections du rapport...[/]")
    latency_interim = (time.monotonic() - t0) * 1000

    sections = [
        _build_executive_summary(all_results),
        _build_market_state(all_results),
        _build_corporate_exposure(all_results),
        _build_strategy_recommendation(all_results),
        _build_position_allocation(all_results),
        _build_audit_summary(all_results, run_id, latency_interim),
    ]

    console.print(f"  [dim]{len(sections)} sections construites[/]")

    # === Etape 2: Narrative Airia (best-effort) ===
    airia_narrative = _generate_airia_narrative(all_results)
    if airia_narrative:
        models_used.append("airia")

    # === Etape 3: Assembler le rapport final ===
    full_text = _assemble_report(sections, run_id, airia_narrative)

    report = ExecutiveReport(
        title=REPORT_TITLE,
        version=REPORT_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
        sections=sections,
        full_text=full_text,
        airia_narrative=airia_narrative or "",
        models_used=models_used,
        total_agents_used=sum(1 for v in all_results.values() if isinstance(v, dict)),
        total_latency_ms=(time.monotonic() - t0) * 1000,
    )

    latency = (time.monotonic() - t0) * 1000

    # === Audit DB ===
    db.save_audit(
        run_id, "report_synthesis", "report_synthesizer",
        input_summary=f"{len(all_results)} sources de resultats",
        output_summary=(
            f"{len(sections)} sections, {len(full_text)} chars, "
            f"airia={'OUI' if airia_narrative else 'NON'}"
        ),
        model_used=", ".join(models_used),
        latency_ms=latency,
    )

    # === Affichage ===
    _print_report(report)

    confidence = 80.0 if len(all_results) >= 5 else 60.0
    if airia_narrative:
        confidence += 10

    console.print(
        f"[green]Report Synthesizer done[/] — {len(sections)} sections, "
        f"{len(full_text)} chars in {int(latency)}ms"
    )

    return StepResult(
        step_name="report_synthesis",
        status=StepStatus.SUCCESS,
        data={
            "report_text": full_text,
            "sections": [
                {
                    "number": s.number,
                    "title": s.title,
                    "content": s.content,
                }
                for s in sections
            ],
            "airia_narrative": report.airia_narrative,
            "total_agents_used": report.total_agents_used,
            "models_used": models_used,
            "report_length": len(full_text),
            "timestamp": report.timestamp,
        },
        confidence=round(min(95, confidence), 1),
        agent_used="report_synthesizer",
        model_used=", ".join(models_used),
        latency_ms=latency,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _print_report(report: ExecutiveReport) -> None:
    """Affiche un apercu du rapport executif."""
    # Table des sections
    table = Table(
        title=f"{report.title} — v{report.version}",
        show_lines=False,
        box=box.ROUNDED,
    )
    table.add_column("#", style="bold", justify="right", width=3)
    table.add_column("Section", style="cyan")
    table.add_column("Taille", justify="right", style="dim")

    for section in report.sections:
        table.add_row(
            str(section.number),
            section.title,
            f"{len(section.content)} chars",
        )

    if report.airia_narrative:
        table.add_row("A", "ANNEXE AIRIA", f"{len(report.airia_narrative)} chars")

    console.print(table)

    # Preview de la premiere section (Resume Executif)
    if report.sections:
        exec_summary = report.sections[0]
        console.print(Panel(
            exec_summary.content,
            title=f"[bold]{exec_summary.title}[/]",
            border_style="cyan",
            width=80,
        ))

    # Panel de synthese
    console.print(Panel(
        f"[bold]Run ID:[/] {report.run_id}\n"
        f"[bold]Timestamp:[/] {report.timestamp}\n"
        f"[bold]Sections:[/] {len(report.sections)}\n"
        f"[bold]Longueur totale:[/] {len(report.full_text):,} caracteres\n"
        f"[bold]Agents utilises:[/] {report.total_agents_used}\n"
        f"[bold]Modeles:[/] {', '.join(report.models_used)}\n"
        f"[bold]Airia:[/] {'OUI — narrative enrichie' if report.airia_narrative else 'NON — rule-based'}",
        title="[bold]Report Synthesizer — Synthese[/]",
        border_style="green" if report.airia_narrative else "yellow",
    ))

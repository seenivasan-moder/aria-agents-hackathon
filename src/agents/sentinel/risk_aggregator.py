"""Agent Sentinel — Risk Aggregator: fusion multi-source des signaux de risque.

Combine les signaux de market_intelligence, corporate_context et sentiment
en un profil de risque unifie avec detection de cascades.

Mode hybride: IA locale M1 (qwen3-30b) + Airia pour cross-validation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.pipeline_engine import StepResult, StepStatus
from src.config import config
from src.services.lm_cluster import query_lm
from src.airia_bridge import bridge
from src import database as db

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Poids de fusion par source
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "market_intelligence": 0.40,
    "corporate_context": 0.30,
    "sentiment": 0.30,
}

# Seuil de detection de cascade: si N sources sont bearish simultanement
CASCADE_THRESHOLD = 2

RISK_ANALYSIS_PROMPT = """\
Tu es un analyste de risque senior. Analyse ces signaux multi-sources et produis
un profil de risque unifie.

SIGNAUX:
{signals_json}

INSTRUCTIONS:
1. Evalue le risque global (0-100)
2. Identifie les risques par classe d'actifs
3. Detecte les cascades de risque (signaux alignes bearish)
4. Propose un niveau d'alerte: LOW / MEDIUM / HIGH / CRITICAL

Reponds EXCLUSIVEMENT en JSON:
{{
  "overall_risk": 65,
  "risk_by_asset_class": {{"crypto": 70, "forex": 55, "commodity": 60}},
  "cascade_detected": true,
  "cascade_description": "BTC et ETH en chute synchronisee avec hausse volatilite forex",
  "alert_level": "HIGH",
  "top_risks": ["BTC volatilite extreme", "EUR/USD exposition non couverte"]
}}"""


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RiskProfile:
    """Profil de risque unifie apres fusion multi-source."""
    overall_risk: float = 0.0
    risk_by_asset_class: dict[str, float] = field(default_factory=dict)
    cascade_detected: bool = False
    cascade_description: str = ""
    alert_level: str = "LOW"
    top_risks: list[str] = field(default_factory=list)
    sources_count: int = 0
    models_used: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# FUSION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _weighted_risk_fusion(
    signals: list[dict],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Fusion ponderee des scores de risque par source."""
    w = weights or DEFAULT_WEIGHTS
    total_weight = 0.0
    weighted_risk = 0.0

    # Risque par classe d'actifs (accumulation ponderee)
    asset_class_risks: dict[str, list[float]] = {}

    for sig in signals:
        source = sig.get("source", "unknown")
        source_weight = w.get(source, 0.2)
        risk = sig.get("risk_score", 0)

        weighted_risk += risk * source_weight
        total_weight += source_weight

        # Accumulation par classe d'actifs
        ac = sig.get("asset_class", "unknown")
        if ac not in asset_class_risks:
            asset_class_risks[ac] = []
        asset_class_risks[ac].append(risk)

    overall = round(weighted_risk / total_weight, 1) if total_weight > 0 else 0

    # Moyenne par classe d'actifs
    risk_by_class = {}
    for ac, scores in asset_class_risks.items():
        risk_by_class[ac] = round(sum(scores) / len(scores), 1)

    return {
        "overall_risk": overall,
        "risk_by_asset_class": risk_by_class,
    }


def _detect_cascade(signals: list[dict]) -> tuple[bool, str]:
    """Detecte les cascades de risque: signaux bearish alignes sur N+ sources."""
    bearish_sources: set[str] = set()
    high_risk_assets: list[str] = []

    for sig in signals:
        direction = sig.get("direction", "neutral")
        risk = sig.get("risk_score", 0)

        if direction == "bearish" or risk > 70:
            bearish_sources.add(sig.get("source", "unknown"))
            if risk > 70:
                high_risk_assets.append(sig.get("symbol", "?"))

    cascade = len(bearish_sources) >= CASCADE_THRESHOLD

    description = ""
    if cascade:
        assets_str = ", ".join(high_risk_assets[:5]) if high_risk_assets else "multiples"
        sources_str = ", ".join(bearish_sources)
        description = (
            f"Cascade detectee: {len(bearish_sources)} sources alignees bearish "
            f"({sources_str}). Actifs a risque: {assets_str}"
        )

    return cascade, description


def _determine_alert_level(overall_risk: float, cascade: bool) -> str:
    """Determine le niveau d'alerte base sur le risque global et la cascade."""
    if cascade and overall_risk > 60:
        return "CRITICAL"
    if overall_risk > 70:
        return "HIGH"
    if overall_risk > 45:
        return "MEDIUM"
    return "LOW"


def _extract_top_risks(signals: list[dict], top_n: int = 5) -> list[str]:
    """Extrait les N risques les plus eleves."""
    scored = []
    for sig in signals:
        risk = sig.get("risk_score", 0)
        symbol = sig.get("symbol", "?")
        source = sig.get("source", "?")
        detail = sig.get("detail", "")
        desc = f"{symbol} ({source}): risque {risk:.0f}"
        if detail:
            desc += f" — {detail}"
        scored.append((risk, desc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [desc for _, desc in scored[:top_n]]


async def _cross_validate_with_m1(signals: list[dict]) -> dict[str, Any] | None:
    """Cross-validation via M1 (qwen3-30b) — enrichissement IA locale."""
    prompt = RISK_ANALYSIS_PROMPT.format(
        signals_json=json.dumps(signals[:20], indent=2, default=str)
    )

    result = await query_lm(
        prompt,
        node_name="M1",
        system="Tu es un analyste de risque quantitatif. Reponds toujours en JSON valide.",
        temperature=0.2,
    )

    if not result.get("ok"):
        console.print(f"  [yellow]M1 cross-validation echouee:[/] {result.get('error', '?')}")
        return None

    # Parser la reponse JSON
    content = result.get("content", "")
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(content[start:end])
            console.print(f"  [dim]M1 cross-validation: {result.get('latency_ms', 0)}ms[/]")
            return parsed
    except (json.JSONDecodeError, ValueError):
        console.print("  [yellow]M1 reponse non parseable[/]")

    return None


def _cross_validate_with_airia(signals: list[dict]) -> dict[str, Any] | None:
    """Cross-validation via Airia — enrichissement cloud."""
    if not bridge.is_available:
        return None

    result = bridge.execute_market_pipeline(signals[:15])
    if result.get("ok") and result.get("parsed"):
        console.print(f"  [dim]Airia cross-validation: {result.get('latency_ms', 0)}ms[/]")
        return result["parsed"]

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT RUN — Point d'entree principal
# ═══════════════════════════════════════════════════════════════════════════════

async def run(
    run_id: str,
    signals: list[dict],
    weights: dict[str, float] | None = None,
) -> StepResult:
    """Execute l'agent Risk Aggregator: fusion multi-source + detection cascade.

    Args:
        run_id: Identifiant du run pipeline
        signals: Liste de signaux provenant de market_intelligence, corporate_context, sentiment
        weights: Poids de fusion par source (optionnel, defaut DEFAULT_WEIGHTS)

    Returns:
        StepResult avec RiskProfile dans data
    """
    t0 = time.monotonic()
    console.print("[bold cyan]Agent Sentinel — Risk Aggregator[/] fusion en cours...")

    models_used = ["rule-based"]

    # === Etape 1: Fusion ponderee locale ===
    fusion = _weighted_risk_fusion(signals, weights)
    overall_risk = fusion["overall_risk"]
    risk_by_class = fusion["risk_by_asset_class"]

    # === Etape 2: Detection de cascade ===
    cascade, cascade_desc = _detect_cascade(signals)

    # === Etape 3: Cross-validation IA (M1 + Airia) ===
    m1_result = await _cross_validate_with_m1(signals)
    if m1_result:
        models_used.append("M1-qwen3-30b")
        # Moyenne entre local et M1 (60% local, 40% M1)
        m1_risk = m1_result.get("overall_risk", overall_risk)
        overall_risk = round(overall_risk * 0.6 + m1_risk * 0.4, 1)

        # Fusionner les risques par classe d'actifs
        m1_by_class = m1_result.get("risk_by_asset_class", {})
        for ac, score in m1_by_class.items():
            if ac in risk_by_class:
                risk_by_class[ac] = round(risk_by_class[ac] * 0.6 + score * 0.4, 1)
            else:
                risk_by_class[ac] = score

        # Cascade detectee par M1 aussi?
        if m1_result.get("cascade_detected") and not cascade:
            cascade = True
            cascade_desc = m1_result.get("cascade_description", "Cascade detectee par M1")

    airia_result = _cross_validate_with_airia(signals)
    if airia_result:
        models_used.append("airia")

    # === Etape 4: Construire le profil de risque final ===
    alert_level = _determine_alert_level(overall_risk, cascade)
    top_risks = _extract_top_risks(signals)

    profile = RiskProfile(
        overall_risk=overall_risk,
        risk_by_asset_class=risk_by_class,
        cascade_detected=cascade,
        cascade_description=cascade_desc,
        alert_level=alert_level,
        top_risks=top_risks,
        sources_count=len(set(s.get("source", "") for s in signals)),
        models_used=models_used,
    )

    latency = (time.monotonic() - t0) * 1000

    # === Audit DB ===
    db.save_audit(
        run_id, "risk_aggregation", "risk_aggregator",
        input_summary=f"{len(signals)} signaux, {profile.sources_count} sources",
        output_summary=(
            f"risque={overall_risk:.0f}, cascade={'OUI' if cascade else 'NON'}, "
            f"alerte={alert_level}, top={len(top_risks)} risques"
        ),
        model_used=", ".join(models_used),
        latency_ms=latency,
    )

    # === Affichage ===
    _print_risk_profile(profile)

    confidence = max(0, 100 - overall_risk * 0.5) if not cascade else max(0, 100 - overall_risk * 0.7)

    console.print(
        f"[green]Risk Aggregator done[/] — risque global: {overall_risk:.0f}, "
        f"alerte: {alert_level}, {profile.sources_count} sources in {int(latency)}ms"
    )

    return StepResult(
        step_name="risk_aggregation",
        status=StepStatus.SUCCESS,
        data={
            "overall_risk": profile.overall_risk,
            "risk_by_asset_class": profile.risk_by_asset_class,
            "cascade_detected": profile.cascade_detected,
            "cascade_description": profile.cascade_description,
            "alert_level": profile.alert_level,
            "top_risks": profile.top_risks,
            "sources_count": profile.sources_count,
            "models_used": profile.models_used,
        },
        confidence=round(confidence, 1),
        agent_used="risk_aggregator",
        model_used=", ".join(models_used),
        latency_ms=latency,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _print_risk_profile(profile: RiskProfile) -> None:
    """Affiche le profil de risque unifie."""
    # Couleur selon le niveau d'alerte
    alert_colors = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red", "CRITICAL": "bold red"}
    alert_color = alert_colors.get(profile.alert_level, "white")

    # Table des risques par classe d'actifs
    table = Table(title="Risk Profile — Par classe d'actifs", show_lines=False)
    table.add_column("Asset Class", style="cyan")
    table.add_column("Risk Score", justify="right")
    table.add_column("Level")

    for ac, score in sorted(profile.risk_by_asset_class.items(), key=lambda x: x[1], reverse=True):
        color = "red" if score > 60 else ("yellow" if score > 35 else "green")
        level = "HIGH" if score > 60 else ("MEDIUM" if score > 35 else "LOW")
        table.add_row(ac, f"[{color}]{score:.0f}[/]", f"[{color}]{level}[/]")

    console.print(table)

    # Panel de synthese
    cascade_str = f"[red]OUI[/] — {profile.cascade_description}" if profile.cascade_detected else "[green]NON[/]"
    console.print(Panel(
        f"[bold]Risque Global:[/] [{alert_color}]{profile.overall_risk:.0f}/100[/]\n"
        f"[bold]Niveau Alerte:[/] [{alert_color}]{profile.alert_level}[/]\n"
        f"[bold]Cascade:[/] {cascade_str}\n"
        f"[bold]Sources:[/] {profile.sources_count}\n"
        f"[bold]Modeles:[/] {', '.join(profile.models_used)}",
        title="[bold]Risk Aggregator — Synthese[/]",
        border_style=alert_color.replace("bold ", ""),
    ))

    # Top risques
    if profile.top_risks:
        console.print("\n[bold]Top Risques:[/]")
        for i, risk in enumerate(profile.top_risks, 1):
            console.print(f"  {i}. {risk}")

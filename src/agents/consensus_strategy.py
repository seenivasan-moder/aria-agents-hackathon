"""Agent 3 — Consensus & Strategy: multi-IA consensus on hedging strategies.

Hybrid mode: LM Studio cluster + Ollama + Airia pipeline — 3-way consensus.
"""

from __future__ import annotations

import json
import time
from typing import Any

from rich.console import Console
from rich.table import Table

from src.config import config
from src.models import (
    MarketSignal, CorporateExposure, ConsensusResult,
    HedgingStrategy, RiskLevel,
)
from src.services.lm_cluster import consensus as cluster_consensus
from src.airia_bridge import bridge
from src import database as db

console = Console()

STRATEGY_PROMPT_TEMPLATE = """You are a senior treasury risk analyst. Based on the following market conditions and corporate exposure, propose exactly 3 hedging strategies: Conservative, Moderate, and Aggressive.

MARKET SIGNALS:
{market_summary}

CORPORATE EXPOSURE:
{exposure_summary}

CONCENTRATION RISKS:
{risks}

For each strategy, provide:
1. Name (Conservative/Moderate/Aggressive)
2. Description (2-3 sentences)
3. Instruments to use (from: FX Forward, FX Option, Commodity Futures, Interest Rate Swap)
4. Estimated cost (% of portfolio)
5. Expected risk reduction (%)
6. Confidence score (0-100)
7. Brief rationale

Respond in JSON format:
{{"strategies": [
  {{"name": "Conservative", "description": "...", "instruments": ["..."], "cost_estimate_pct": 0.5, "risk_reduction_pct": 80, "confidence": 85, "rationale": "..."}},
  {{"name": "Moderate", "description": "...", "instruments": ["..."], "cost_estimate_pct": 0.3, "risk_reduction_pct": 60, "confidence": 75, "rationale": "..."}},
  {{"name": "Aggressive", "description": "...", "instruments": ["..."], "cost_estimate_pct": 0.15, "risk_reduction_pct": 30, "confidence": 60, "rationale": "..."}}
]}}"""


def _build_market_summary(signals: list[MarketSignal]) -> str:
    lines = []
    for s in signals[:10]:
        lines.append(f"- {s.symbol} ({s.asset_class}): {s.price:.4f}, 24h={s.change_24h:+.2f}%, vol={s.volatility:.2f}%, risk={s.risk_score:.0f}, dir={s.direction.value}")
    return "\n".join(lines)


def _build_exposure_summary(exposure: CorporateExposure) -> str:
    lines = [f"Company: {exposure.company_id}, Base: {exposure.base_currency}"]
    lines.append(f"Total Assets: {exposure.total_assets:,.0f}, Net Cash: {exposure.net_cash:,.0f}")
    for p in exposure.positions:
        lines.append(f"- {p.currency}: net={p.net_exposure:+,.0f} ({p.exposure_pct:.1f}%)")
    return "\n".join(lines)


def _parse_strategies(response_text: str) -> list[HedgingStrategy]:
    """Parse LLM response into HedgingStrategy objects."""
    text = response_text.strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return []

    try:
        data = json.loads(text[start:end])
        strategies = []
        for s in data.get("strategies", []):
            risk_map = {"conservative": RiskLevel.LOW, "moderate": RiskLevel.MEDIUM, "aggressive": RiskLevel.HIGH}
            strategies.append(HedgingStrategy(
                name=s.get("name", "Unknown"),
                description=s.get("description", ""),
                instruments=s.get("instruments", []),
                cost_estimate_pct=float(s.get("cost_estimate_pct", 0)),
                risk_reduction_pct=float(s.get("risk_reduction_pct", 0)),
                confidence=float(s.get("confidence", 50)),
                rationale=s.get("rationale", ""),
                risk_level=risk_map.get(s.get("name", "").lower(), RiskLevel.MEDIUM),
            ))
        return strategies
    except (json.JSONDecodeError, KeyError, ValueError):
        return []


def _fallback_strategies() -> list[HedgingStrategy]:
    """Return default strategies if all IA sources fail."""
    return [
        HedgingStrategy(
            name="Conservative",
            description="Full hedge on all major currency exposures using forwards. Commodity futures for 6-month coverage.",
            instruments=["FX Forward", "Commodity Futures"],
            cost_estimate_pct=0.45, risk_reduction_pct=85, confidence=80,
            rationale="Maximum protection against currency and commodity volatility.",
            risk_level=RiskLevel.LOW,
        ),
        HedgingStrategy(
            name="Moderate",
            description="Selective hedging on top 3 exposures with options for flexibility. Partial commodity coverage.",
            instruments=["FX Option", "FX Forward", "Commodity Futures"],
            cost_estimate_pct=0.25, risk_reduction_pct=60, confidence=72,
            rationale="Balanced approach protecting major risks while maintaining upside potential.",
            risk_level=RiskLevel.MEDIUM,
        ),
        HedgingStrategy(
            name="Aggressive",
            description="Minimal hedging — only JPY and CNY forwards for largest exposures. No commodity hedge.",
            instruments=["FX Forward"],
            cost_estimate_pct=0.10, risk_reduction_pct=30, confidence=55,
            rationale="Low cost, accepts significant market risk for potential gains.",
            risk_level=RiskLevel.HIGH,
        ),
    ]


async def run(
    run_id: str,
    signals: list[MarketSignal],
    exposure: CorporateExposure,
) -> ConsensusResult:
    """Execute Agent 3: 3-way consensus (LM Studio + Ollama + Airia)."""
    t0 = time.monotonic()
    console.print("[bold cyan]Agent 3 — Consensus & Strategy[/] generating...")

    market_summary = _build_market_summary(signals)
    exposure_summary = _build_exposure_summary(exposure)
    risks = "\n".join(f"- {r}" for r in exposure.concentration_risks) or "No critical concentration risks."

    prompt = STRATEGY_PROMPT_TEMPLATE.format(
        market_summary=market_summary,
        exposure_summary=exposure_summary,
        risks=risks,
    )

    # === 3-way consensus: Local cluster + Airia ===
    all_strategies: list[list[HedgingStrategy]] = []
    models_used: list[str] = []

    # Source 1+2: LM Studio (M1) + Ollama (OL1) in parallel
    cluster_result = await cluster_consensus(prompt, nodes=["M1", "OL1"])
    for resp in cluster_result.get("responses", []):
        if resp.get("ok"):
            parsed = _parse_strategies(resp["content"])
            if parsed:
                all_strategies.append(parsed)
                models_used.append(resp.get("model", resp.get("node", "local")))

    # Source 3: Airia pipeline (best-effort)
    if bridge.is_available:
        airia_result = bridge.execute_consensus_pipeline(
            [s.model_dump(mode="json") for s in signals[:10]],
            exposure.model_dump(mode="json"),
        )
        if airia_result.get("ok") and airia_result.get("parsed"):
            parsed_data = airia_result["parsed"]
            if isinstance(parsed_data, dict):
                airia_strats = _parse_strategies(json.dumps(parsed_data))
                if airia_strats:
                    all_strategies.append(airia_strats)
                    models_used.append("airia-pipeline")

    # === Merge consensus ===
    if all_strategies:
        # Use first complete set as base
        strategies = all_strategies[0]

        # Average confidence scores across all sources
        if len(all_strategies) > 1:
            for i, s in enumerate(strategies):
                confs = []
                for source in all_strategies:
                    if i < len(source):
                        confs.append(source[i].confidence)
                if confs:
                    s.confidence = round(sum(confs) / len(confs), 1)

            # Check for dissenting views (>20% confidence spread)
            dissenting = []
            for i, s in enumerate(strategies):
                confs = [src[i].confidence for src in all_strategies if i < len(src)]
                if confs and max(confs) - min(confs) > 20:
                    dissenting.append(f"{s.name}: confidence spread {min(confs):.0f}-{max(confs):.0f}")
    else:
        strategies = _fallback_strategies()
        models_used = ["fallback"]

    # Build result
    recommended = max(range(len(strategies)), key=lambda i: strategies[i].confidence) if strategies else 0
    consensus_score = round(sum(s.confidence for s in strategies) / len(strategies), 1) if strategies else 0

    result = ConsensusResult(
        strategies=strategies,
        recommended_index=recommended,
        consensus_score=consensus_score,
        dissenting_views=dissenting if all_strategies and len(all_strategies) > 1 else [],
        models_used=models_used,
    )

    latency = (time.monotonic() - t0) * 1000

    # Save to DB
    db.save_strategies(
        run_id,
        [s.model_dump(mode="json") for s in strategies],
        recommended_idx=recommended,
    )
    db.save_audit(
        run_id, "consensus_strategy", "consensus_strategy",
        input_summary=f"{len(signals)} signals + {exposure.company_id} exposure",
        output_summary=f"{len(strategies)} strategies, recommended: {strategies[recommended].name}, "
                       f"consensus={consensus_score}, sources={len(all_strategies)}",
        model_used=", ".join(models_used),
        latency_ms=latency,
    )

    _print_strategies(result)
    console.print(f"[green]Agent 3 done[/] — consensus score: {consensus_score} "
                  f"({len(all_strategies)} sources) in {int(latency)}ms")
    return result


def _print_strategies(result: ConsensusResult) -> None:
    table = Table(title="Hedging Strategies", show_lines=True)
    table.add_column("#", style="bold", width=3)
    table.add_column("Strategy", style="cyan")
    table.add_column("Risk Level")
    table.add_column("Cost %", justify="right")
    table.add_column("Risk Red. %", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Instruments")

    for i, s in enumerate(result.strategies):
        rec = " *" if i == result.recommended_index else ""
        risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(s.risk_level.value, "white")
        table.add_row(
            f"{i+1}{rec}",
            s.name,
            f"[{risk_color}]{s.risk_level.value}[/]",
            f"{s.cost_estimate_pct:.2f}%",
            f"{s.risk_reduction_pct:.0f}%",
            f"{s.confidence:.0f}",
            ", ".join(s.instruments),
        )

    console.print(table)

    if result.dissenting_views:
        console.print("\n[yellow]Dissenting views:[/]")
        for d in result.dissenting_views:
            console.print(f"  [yellow]![/] {d}")

    console.print(f"\nModels used: {', '.join(result.models_used)}")
    console.print(f"Consensus score: {result.consensus_score:.1f}")

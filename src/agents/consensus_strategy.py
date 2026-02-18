"""Agent 3 — Consensus & Strategy: multi-IA consensus on hedging strategies.

Adapted from JARVIS consensus() pattern — queries M1 + OL1 + Airia in parallel.
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
    # Try to extract JSON from response
    text = response_text.strip()

    # Find JSON block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return _fallback_strategies()

    try:
        data = json.loads(text[start:end])
        strategies = []
        for s in data.get("strategies", []):
            risk_map = {"conservative": RiskLevel.LOW, "moderate": RiskLevel.MEDIUM, "aggressive": RiskLevel.HIGH}
            strategies.append(HedgingStrategy(
                name=s.get("name", "Unknown"),
                description=s.get("description", ""),
                instruments=s.get("instruments", []),
                cost_estimate_pct=s.get("cost_estimate_pct", 0),
                risk_reduction_pct=s.get("risk_reduction_pct", 0),
                confidence=s.get("confidence", 50),
                rationale=s.get("rationale", ""),
                risk_level=risk_map.get(s.get("name", "").lower(), RiskLevel.MEDIUM),
            ))
        return strategies if strategies else _fallback_strategies()
    except (json.JSONDecodeError, KeyError):
        return _fallback_strategies()


def _fallback_strategies() -> list[HedgingStrategy]:
    """Return default strategies if LLM parsing fails."""
    return [
        HedgingStrategy(
            name="Conservative",
            description="Full hedge on all major currency exposures using forwards. Commodity futures for 6-month coverage.",
            instruments=["FX Forward", "Commodity Futures"],
            cost_estimate_pct=0.45,
            risk_reduction_pct=85,
            confidence=80,
            rationale="Maximum protection against currency and commodity volatility.",
            risk_level=RiskLevel.LOW,
        ),
        HedgingStrategy(
            name="Moderate",
            description="Selective hedging on top 3 exposures with options for flexibility. Partial commodity coverage.",
            instruments=["FX Option", "FX Forward", "Commodity Futures"],
            cost_estimate_pct=0.25,
            risk_reduction_pct=60,
            confidence=72,
            rationale="Balanced approach protecting major risks while maintaining upside potential.",
            risk_level=RiskLevel.MEDIUM,
        ),
        HedgingStrategy(
            name="Aggressive",
            description="Minimal hedging — only JPY and CNY forwards for largest exposures. No commodity hedge.",
            instruments=["FX Forward"],
            cost_estimate_pct=0.10,
            risk_reduction_pct=30,
            confidence=55,
            rationale="Low cost, accepts significant market risk for potential gains.",
            risk_level=RiskLevel.HIGH,
        ),
    ]


async def run(
    run_id: str,
    signals: list[MarketSignal],
    exposure: CorporateExposure,
) -> ConsensusResult:
    """Execute Agent 3: generate consensus hedging strategies."""
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

    # Query multiple nodes in parallel (consensus pattern)
    consensus_result = await cluster_consensus(prompt, nodes=["M1", "OL1"])

    # Parse all responses and pick best strategies
    all_strategies: list[list[HedgingStrategy]] = []
    models_used = consensus_result.get("models_used", [])

    for resp in consensus_result.get("responses", []):
        if resp.get("ok"):
            strategies = _parse_strategies(resp["content"])
            all_strategies.append(strategies)

    # Merge: use first successful parse, or fallback
    if all_strategies:
        strategies = all_strategies[0]
        # Average confidence across multiple responses
        if len(all_strategies) > 1:
            for i, s in enumerate(strategies):
                confs = [all_strategies[j][i].confidence for j in range(len(all_strategies)) if i < len(all_strategies[j])]
                if confs:
                    s.confidence = round(sum(confs) / len(confs), 1)
    else:
        strategies = _fallback_strategies()
        models_used = ["fallback"]

    # Build consensus result
    recommended = max(range(len(strategies)), key=lambda i: strategies[i].confidence) if strategies else 0
    consensus_score = round(sum(s.confidence for s in strategies) / len(strategies), 1) if strategies else 0

    result = ConsensusResult(
        strategies=strategies,
        recommended_index=recommended,
        consensus_score=consensus_score,
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
        output_summary=f"{len(strategies)} strategies, recommended: {strategies[recommended].name}, consensus={consensus_score}",
        model_used=", ".join(models_used),
        latency_ms=latency,
    )

    # Display
    _print_strategies(result)

    console.print(f"[green]Agent 3 done[/] — consensus score: {consensus_score} in {int(latency)}ms")
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
    console.print(f"\nModels used: {', '.join(result.models_used)}")
    console.print(f"Consensus score: {result.consensus_score:.1f}")

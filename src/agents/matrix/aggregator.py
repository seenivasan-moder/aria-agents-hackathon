"""Agent Aggregator — Fuses results from parallel vectorial execution.

Combines outputs from multiple AI providers using weighted scoring,
consensus detection, and conflict resolution strategies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.table import Table

from src.pipeline_engine import StepResult, StepStatus
from src import database as db

console = Console()


@dataclass
class AggregationStrategy:
    name: str
    method: str  # "weighted_avg", "majority_vote", "best_confidence", "union", "intersection"
    weights: dict[str, float] | None = None


# ── Pre-built strategies ─────────────────────────────────────────────

CONSENSUS_STRATEGY = AggregationStrategy(
    "consensus", "weighted_avg",
    weights={"M1": 0.45, "OL1": 0.20, "Airia": 0.35},
)

BEST_WINS_STRATEGY = AggregationStrategy("best_wins", "best_confidence")

UNION_STRATEGY = AggregationStrategy("union", "union")


def aggregate(results: list[StepResult], strategy: AggregationStrategy) -> StepResult:
    """Aggregate multiple step results using the specified strategy."""
    successful = [r for r in results if r.status == StepStatus.SUCCESS]

    if not successful:
        return StepResult(
            step_name=f"aggregator_{strategy.name}",
            status=StepStatus.FAILED,
            error="No successful results to aggregate",
            agent_used="aggregator",
        )

    if strategy.method == "weighted_avg":
        return _weighted_average(successful, strategy)
    elif strategy.method == "best_confidence":
        return _best_confidence(successful, strategy)
    elif strategy.method == "union":
        return _union_merge(successful, strategy)
    elif strategy.method == "majority_vote":
        return _majority_vote(successful, strategy)
    else:
        return _best_confidence(successful, strategy)


def _weighted_average(results: list[StepResult], strategy: AggregationStrategy) -> StepResult:
    """Weighted average of confidence scores."""
    weights = strategy.weights or {}
    total_weight = 0
    weighted_conf = 0

    for r in results:
        w = weights.get(r.agent_used, 1.0)
        weighted_conf += r.confidence * w
        total_weight += w

    avg_conf = weighted_conf / total_weight if total_weight else 0

    # Detect consensus vs dissent
    confidences = [r.confidence for r in results]
    spread = max(confidences) - min(confidences) if len(confidences) > 1 else 0
    has_dissent = spread > 20

    return StepResult(
        step_name=f"aggregator_{strategy.name}",
        status=StepStatus.SUCCESS,
        data={
            "sources": len(results),
            "agents": [r.agent_used for r in results],
            "data": [r.data for r in results],
            "spread": spread,
            "dissent": has_dissent,
        },
        confidence=avg_conf,
        agent_used="aggregator",
    )


def _best_confidence(results: list[StepResult], strategy: AggregationStrategy) -> StepResult:
    """Pick the result with highest confidence."""
    best = max(results, key=lambda r: r.confidence)
    return StepResult(
        step_name=f"aggregator_{strategy.name}",
        status=StepStatus.SUCCESS,
        data=best.data,
        confidence=best.confidence,
        agent_used=f"aggregator(best={best.agent_used})",
        model_used=best.model_used,
    )


def _union_merge(results: list[StepResult], strategy: AggregationStrategy) -> StepResult:
    """Merge all data (union of results)."""
    merged = []
    for r in results:
        if isinstance(r.data, list):
            merged.extend(r.data)
        elif r.data is not None:
            merged.append(r.data)

    avg_conf = sum(r.confidence for r in results) / len(results) if results else 0

    return StepResult(
        step_name=f"aggregator_{strategy.name}",
        status=StepStatus.SUCCESS,
        data=merged,
        confidence=avg_conf,
        agent_used="aggregator",
    )


def _majority_vote(results: list[StepResult], strategy: AggregationStrategy) -> StepResult:
    """Majority voting on data values."""
    # Count occurrences of each unique result
    votes: dict[str, int] = {}
    for r in results:
        key = str(r.data)[:100] if r.data else "none"
        votes[key] = votes.get(key, 0) + 1

    winner = max(votes.items(), key=lambda x: x[1])
    winner_result = next(r for r in results if str(r.data)[:100] == winner[0])

    return StepResult(
        step_name=f"aggregator_{strategy.name}",
        status=StepStatus.SUCCESS,
        data=winner_result.data,
        confidence=winner[1] / len(results) * 100,
        agent_used=f"aggregator(vote={winner[1]}/{len(results)})",
    )


async def run(run_id: str, results: list[StepResult],
              strategy: AggregationStrategy | None = None,
              step_name: str = "aggregation") -> StepResult:
    """Execute Agent Aggregator: fuse vectorial results."""
    t0 = time.monotonic()
    console.print(f"[bold magenta]Agent Aggregator[/] fusing {len(results)} results...")

    if strategy is None:
        strategy = CONSENSUS_STRATEGY

    aggregated = aggregate(results, strategy)
    aggregated.latency_ms = (time.monotonic() - t0) * 1000
    aggregated.step_name = step_name

    db.save_audit(
        run_id, "aggregation", "aggregator",
        input_summary=f"{len(results)} results, strategy={strategy.name}",
        output_summary=f"confidence={aggregated.confidence:.0f}%, "
                       f"sources={len(results)}",
        model_used=strategy.name,
        latency_ms=aggregated.latency_ms,
    )

    console.print(f"[green]Aggregator done[/] — {strategy.name}, "
                  f"confidence={aggregated.confidence:.0f}% in {int(aggregated.latency_ms)}ms")

    return aggregated

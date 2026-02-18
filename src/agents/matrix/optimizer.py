"""Agent Optimizer — Real-time routing optimization for the matrix pipeline.

Dynamically adjusts agent routing based on:
- Historical latency data
- Current node health
- Task complexity estimation
- Cost optimization (prefer local over cloud)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

from src.pipeline_engine import StepResult, StepStatus
from src import database as db

console = Console()


@dataclass
class NodeProfile:
    """Performance profile for an AI node."""
    name: str
    backend: str  # lm_studio, ollama, airia, local
    avg_latency_ms: float = 0
    success_rate: float = 100
    cost_factor: float = 0  # 0=free (local), 1=cloud
    capacity: str = "high"  # high, medium, low
    specialties: list[str] = field(default_factory=list)

    @property
    def efficiency_score(self) -> float:
        """Combined efficiency score (higher is better)."""
        latency_score = max(0, 100 - self.avg_latency_ms / 10)  # < 1s = high
        cost_score = (1 - self.cost_factor) * 100  # free = 100
        reliability = self.success_rate
        return (latency_score * 0.3 + cost_score * 0.3 + reliability * 0.4)


# ── Known node profiles ──────────────────────────────────────────────

NODE_PROFILES = {
    "M1": NodeProfile(
        "M1", "lm_studio",
        avg_latency_ms=500, success_rate=95, cost_factor=0,
        capacity="high",
        specialties=["complex", "analysis", "trading", "code"],
    ),
    "OL1": NodeProfile(
        "OL1", "ollama",
        avg_latency_ms=200, success_rate=90, cost_factor=0,
        capacity="medium",
        specialties=["simple", "quick", "chat", "correction"],
    ),
    "Airia": NodeProfile(
        "Airia", "airia",
        avg_latency_ms=2000, success_rate=80, cost_factor=0.8,
        capacity="high",
        specialties=["complex", "analysis", "enrichment", "summary"],
    ),
    "local": NodeProfile(
        "local", "local",
        avg_latency_ms=1, success_rate=100, cost_factor=0,
        capacity="high",
        specialties=["system", "file", "process"],
    ),
}


@dataclass
class RoutingDecision:
    """Optimizer's routing recommendation."""
    recommended_node: str
    score: float
    reason: str
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    estimated_latency_ms: float = 0


def optimize_routing(task_type: str, complexity: str = "medium",
                     prefer_local: bool = True) -> RoutingDecision:
    """Determine the optimal node for a given task.

    Args:
        task_type: Type of task (analysis, trading, system, chat, etc.)
        complexity: Task complexity (simple, medium, complex)
        prefer_local: Prefer local nodes over cloud (cost optimization)
    """
    scores: list[tuple[str, float, str]] = []

    for name, profile in NODE_PROFILES.items():
        score = profile.efficiency_score

        # Specialty bonus
        if task_type in profile.specialties:
            score += 20

        # Complexity matching
        if complexity == "complex" and profile.capacity == "high":
            score += 15
        elif complexity == "simple" and profile.capacity in ("medium", "high"):
            score += 10

        # Local preference
        if prefer_local and profile.cost_factor == 0:
            score += 10
        elif not prefer_local and profile.cost_factor > 0:
            score += 5  # slight cloud bonus when preferred

        # Penalty for low success rate
        if profile.success_rate < 80:
            score -= 15

        reason = f"specialty={task_type in profile.specialties}, "
        reason += f"efficiency={profile.efficiency_score:.0f}, "
        reason += f"cost={profile.cost_factor}"

        scores.append((name, min(score, 100), reason))

    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)
    best = scores[0]

    return RoutingDecision(
        recommended_node=best[0],
        score=best[1],
        reason=best[2],
        alternatives=[(s[0], s[1]) for s in scores[1:]],
        estimated_latency_ms=NODE_PROFILES[best[0]].avg_latency_ms,
    )


async def run(run_id: str, input_data: Any = None) -> StepResult:
    """Execute Agent Optimizer: analyze and recommend optimal routing."""
    t0 = time.monotonic()
    console.print("[bold magenta]Agent Optimizer[/] analyzing node performance...")

    # Display current node profiles
    table = Table(title="Node Performance Profiles")
    table.add_column("Node", style="cyan")
    table.add_column("Backend", style="dim")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Success Rate", justify="right")
    table.add_column("Cost", justify="center")
    table.add_column("Efficiency", justify="right", style="bold")
    table.add_column("Specialties", style="dim")

    for name, profile in NODE_PROFILES.items():
        eff = profile.efficiency_score
        color = "green" if eff >= 70 else "yellow" if eff >= 50 else "red"
        table.add_row(
            name,
            profile.backend,
            f"{profile.avg_latency_ms:.0f}ms",
            f"{profile.success_rate:.0f}%",
            "[green]FREE[/]" if profile.cost_factor == 0 else f"[yellow]{profile.cost_factor:.1f}[/]",
            f"[{color}]{eff:.0f}[/]",
            ", ".join(profile.specialties[:4]),
        )

    console.print(table)

    # Generate routing recommendations for common tasks
    task_types = ["analysis", "trading", "system", "chat", "code"]
    recommendations = {}
    for task in task_types:
        decision = optimize_routing(task)
        recommendations[task] = decision

    latency = (time.monotonic() - t0) * 1000

    db.save_audit(
        run_id, "optimization", "optimizer",
        input_summary=f"{len(NODE_PROFILES)} nodes profiled",
        output_summary=f"{len(recommendations)} routing decisions generated",
        model_used="heuristic",
        latency_ms=latency,
    )

    console.print(f"[green]Optimizer done[/] — {len(recommendations)} routes optimized in {int(latency)}ms")

    return StepResult(
        step_name="optimization",
        status=StepStatus.SUCCESS,
        data=recommendations,
        confidence=90,
        agent_used="optimizer",
        latency_ms=latency,
    )

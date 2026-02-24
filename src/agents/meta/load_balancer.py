"""Agent Load-Balancer — Dynamic weighted round-robin with health-aware routing.

Tracks real-time latency metrics per node and implements intelligent
load distribution across the AI cluster:
- Weighted round-robin based on node capacity and health
- Automatic degradation detection (latency > 2x average triggers reroute)
- Real-time health dashboard with rich tables
- Node affinity for task-type specialization

Supported nodes:
- M1 (LM Studio): Deep analysis, complex reasoning (127.0.0.1:1234)
- OL1 (Ollama): Quick answers, corrections (127.0.0.1:11434)
- Airia: Pipeline orchestration, formal analysis (cloud)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rich.console import Console
from rich.table import Table

from src.pipeline_engine import StepResult, StepStatus
from src.config import config
from src.utils.http_pool import get_avg_latency, get_all_metrics
from src import database as db

console = Console()


class NodeStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


class TaskType(str, Enum):
    ANALYSIS = "analysis"
    TRADING = "trading"
    CODE = "code"
    CHAT = "chat"
    SYSTEM = "system"
    ENRICHMENT = "enrichment"
    COMPLIANCE = "compliance"


@dataclass
class NodeHealth:
    """Real-time health metrics for a single node."""
    name: str
    latency_ms: float = 0.0
    error_count: int = 0
    success_count: int = 0
    last_success: float = 0.0       # monotonic timestamp
    last_failure: float = 0.0       # monotonic timestamp
    capacity: float = 1.0           # 0.0 (saturated) to 1.0 (idle)
    weight: float = 1.0             # routing weight (adjusted dynamically)
    status: NodeStatus = NodeStatus.HEALTHY
    specialties: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.error_count
        return (self.success_count / total * 100) if total > 0 else 100.0

    @property
    def effective_weight(self) -> float:
        """Weight adjusted by health status and capacity."""
        if self.status == NodeStatus.DOWN:
            return 0.0
        modifier = 0.5 if self.status == NodeStatus.DEGRADED else 1.0
        return self.weight * self.capacity * modifier


# ── Node Health Registry ──────────────────────────────────────────────

NODE_HEALTH: dict[str, NodeHealth] = {
    "M1": NodeHealth(
        name="M1", weight=1.0, capacity=1.0,
        specialties=["analysis", "trading", "code", "compliance"],
    ),
    "OL1": NodeHealth(
        name="OL1", weight=0.6, capacity=1.0,
        specialties=["chat", "system", "enrichment"],
    ),
    "Airia": NodeHealth(
        name="Airia", weight=0.8, capacity=1.0,
        specialties=["analysis", "enrichment", "compliance"],
    ),
}

# ── Round-robin state ─────────────────────────────────────────────────

_rr_index: int = 0


def _sync_metrics() -> None:
    """Sync latency metrics from http_pool into NODE_HEALTH."""
    all_metrics = get_all_metrics()
    for name, health in NODE_HEALTH.items():
        metrics = all_metrics.get(name)
        if metrics:
            health.latency_ms = metrics["avg_ms"]


def _detect_degradation() -> None:
    """Detect degraded nodes: latency > 2x global average triggers DEGRADED status."""
    latencies = [h.latency_ms for h in NODE_HEALTH.values() if h.latency_ms > 0]
    if not latencies:
        return

    global_avg = sum(latencies) / len(latencies)
    threshold = global_avg * 2

    for health in NODE_HEALTH.values():
        if health.latency_ms <= 0:
            continue

        if health.error_count >= 3 and health.success_count == 0:
            health.status = NodeStatus.DOWN
        elif health.latency_ms > threshold:
            health.status = NodeStatus.DEGRADED
            console.print(
                f"  [yellow]Load-Balancer: {health.name} degraded[/] "
                f"(latency {health.latency_ms:.0f}ms > threshold {threshold:.0f}ms)"
            )
        else:
            health.status = NodeStatus.HEALTHY


def record_success(node: str, latency_ms: float) -> None:
    """Record a successful request to a node."""
    health = NODE_HEALTH.get(node)
    if not health:
        return
    health.success_count += 1
    health.last_success = time.monotonic()
    health.latency_ms = (health.latency_ms * 0.7) + (latency_ms * 0.3)  # EMA
    _detect_degradation()


def record_failure(node: str) -> None:
    """Record a failed request to a node."""
    health = NODE_HEALTH.get(node)
    if not health:
        return
    health.error_count += 1
    health.last_failure = time.monotonic()
    _detect_degradation()


def optimize(task_type: str) -> str:
    """Return the best node name for a given task type.

    Uses weighted round-robin among eligible nodes, with specialty bonus
    and health-aware filtering.
    """
    global _rr_index

    _sync_metrics()
    _detect_degradation()

    # Build scored candidates
    candidates: list[tuple[str, float]] = []

    for name, health in NODE_HEALTH.items():
        if health.status == NodeStatus.DOWN:
            continue

        score = health.effective_weight

        # Specialty bonus: +50% if node specializes in this task type
        if task_type in health.specialties:
            score *= 1.5

        # Latency penalty: prefer lower latency
        if health.latency_ms > 0:
            latency_penalty = max(0, 1 - health.latency_ms / 5000)
            score *= (0.6 + 0.4 * latency_penalty)

        # Success rate bonus
        score *= (health.success_rate / 100)

        candidates.append((name, score))

    if not candidates:
        return "M1"  # ultimate fallback

    # Sort by score descending
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Weighted round-robin among top candidates
    total_score = sum(s for _, s in candidates)
    if total_score == 0:
        return candidates[0][0]

    # Distribute based on weight proportions
    _rr_index = (_rr_index + 1) % max(len(candidates), 1)

    # If top candidate has >2x the score of runner-up, always pick it
    if len(candidates) >= 2 and candidates[0][1] > candidates[1][1] * 2:
        return candidates[0][0]

    # Otherwise use round-robin index mapped to weighted selection
    cumulative = 0.0
    target = (_rr_index / max(len(candidates), 1)) * total_score
    for name, score in candidates:
        cumulative += score
        if cumulative >= target:
            return name

    return candidates[0][0]


def get_health_snapshot() -> dict[str, dict[str, Any]]:
    """Return a snapshot of all node health data."""
    _sync_metrics()
    _detect_degradation()
    snapshot = {}
    for name, h in NODE_HEALTH.items():
        snapshot[name] = {
            "status": h.status.value,
            "latency_ms": round(h.latency_ms, 1),
            "success_rate": round(h.success_rate, 1),
            "error_count": h.error_count,
            "capacity": round(h.capacity, 2),
            "effective_weight": round(h.effective_weight, 3),
            "specialties": h.specialties,
        }
    return snapshot


def _print_health_dashboard() -> None:
    """Print a rich health dashboard table."""
    table = Table(title="Load Balancer — Node Health Dashboard", show_lines=True)
    table.add_column("Node", style="cyan", width=8)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Latency", justify="right", width=10)
    table.add_column("Success Rate", justify="right", width=12)
    table.add_column("Errors", justify="right", width=7)
    table.add_column("Capacity", justify="right", width=10)
    table.add_column("Eff. Weight", justify="right", width=11)
    table.add_column("Specialties", style="dim")

    for name, h in NODE_HEALTH.items():
        status_color = {
            NodeStatus.HEALTHY: "green",
            NodeStatus.DEGRADED: "yellow",
            NodeStatus.DOWN: "red",
        }[h.status]

        rate = h.success_rate
        rate_color = "green" if rate >= 90 else "yellow" if rate >= 70 else "red"

        table.add_row(
            name,
            f"[{status_color}]{h.status.value}[/]",
            f"{h.latency_ms:.0f}ms" if h.latency_ms > 0 else "[dim]--[/]",
            f"[{rate_color}]{rate:.0f}%[/]",
            str(h.error_count),
            f"{h.capacity:.0%}",
            f"{h.effective_weight:.3f}",
            ", ".join(h.specialties[:4]),
        )

    console.print(table)

    # Routing recommendations
    rec_table = Table(title="Routing Recommendations", show_lines=False)
    rec_table.add_column("Task Type", style="cyan")
    rec_table.add_column("Best Node", style="bold")

    for task in TaskType:
        best = optimize(task.value)
        rec_table.add_row(task.value, f"[green]{best}[/]")

    console.print(rec_table)


async def run(run_id: str, input_data: Any = None) -> StepResult:
    """Execute Agent Load-Balancer: analyze cluster health and optimize routing."""
    t0 = time.monotonic()
    console.print("[bold magenta]Agent Load-Balancer[/] analyzing cluster health...")

    _sync_metrics()
    _detect_degradation()
    _print_health_dashboard()

    # Generate routing map for all task types
    routing_map = {}
    for task in TaskType:
        routing_map[task.value] = optimize(task.value)

    snapshot = get_health_snapshot()
    latency = (time.monotonic() - t0) * 1000

    # Count healthy vs degraded
    healthy = sum(1 for h in NODE_HEALTH.values() if h.status == NodeStatus.HEALTHY)
    degraded = sum(1 for h in NODE_HEALTH.values() if h.status == NodeStatus.DEGRADED)
    down = sum(1 for h in NODE_HEALTH.values() if h.status == NodeStatus.DOWN)

    db.save_audit(
        run_id, "load_balancing", "load_balancer",
        input_summary=f"{len(NODE_HEALTH)} nodes monitored",
        output_summary=f"healthy={healthy}, degraded={degraded}, down={down}, "
                       f"routes={len(routing_map)}",
        model_used="weighted-round-robin",
        latency_ms=latency,
    )

    console.print(
        f"[green]Load-Balancer done[/] — {healthy} healthy, "
        f"{degraded} degraded, {down} down in {int(latency)}ms"
    )

    return StepResult(
        step_name="load_balancing",
        status=StepStatus.SUCCESS,
        data={
            "routing_map": routing_map,
            "health_snapshot": snapshot,
            "summary": {
                "healthy": healthy,
                "degraded": degraded,
                "down": down,
                "total": len(NODE_HEALTH),
            },
        },
        confidence=95 if down == 0 else (70 if healthy > 0 else 30),
        agent_used="load_balancer",
        latency_ms=latency,
    )

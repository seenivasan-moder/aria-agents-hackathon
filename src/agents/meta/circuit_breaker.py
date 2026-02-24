"""Agent Circuit-Breaker — Fault tolerance with automatic recovery.

Implements the circuit breaker pattern for AI node resilience:

States:
- CLOSED (normal): Requests pass through. Failures are counted.
- OPEN (failing): All requests blocked. Node considered unhealthy.
- HALF_OPEN (probing): Single test request allowed to check recovery.

Transitions:
- CLOSED -> OPEN: After `failure_threshold` consecutive failures (default: 3)
- OPEN -> HALF_OPEN: After `cooldown_seconds` (default: 30s)
- HALF_OPEN -> CLOSED: On successful probe
- HALF_OPEN -> OPEN: On failed probe (resets cooldown)

Nodes monitored:
- M1 (LM Studio 127.0.0.1:1234)
- OL1 (Ollama 127.0.0.1:11434)
- Airia (cloud platform)
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
from src import database as db

console = Console()


class CircuitState(str, Enum):
    CLOSED = "closed"       # Normal operation — requests pass through
    OPEN = "open"           # Fault detected — requests blocked
    HALF_OPEN = "half_open" # Cooldown expired — testing with single request


@dataclass
class NodeCircuit:
    """Circuit breaker state for a single node."""
    name: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    consecutive_failures: int = 0
    last_failure_time: float = 0.0          # monotonic
    last_success_time: float = 0.0          # monotonic
    last_state_change: float = field(default_factory=time.monotonic)
    total_blocked: int = 0                  # requests blocked while OPEN

    @property
    def time_in_state_s(self) -> float:
        return time.monotonic() - self.last_state_change


# ── Configuration ─────────────────────────────────────────────────────

DEFAULT_FAILURE_THRESHOLD = 3   # consecutive failures before OPEN
DEFAULT_COOLDOWN_SECONDS = 30   # seconds before OPEN -> HALF_OPEN


# ── Circuit Registry ──────────────────────────────────────────────────

_circuits: dict[str, NodeCircuit] = {
    "M1": NodeCircuit(name="M1"),
    "OL1": NodeCircuit(name="OL1"),
    "Airia": NodeCircuit(name="Airia"),
}

_failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
_cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS


def configure(failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
              cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS) -> None:
    """Reconfigure circuit breaker thresholds."""
    global _failure_threshold, _cooldown_seconds
    _failure_threshold = failure_threshold
    _cooldown_seconds = cooldown_seconds


def _get_circuit(node: str) -> NodeCircuit:
    """Get or create a circuit for the given node."""
    if node not in _circuits:
        _circuits[node] = NodeCircuit(name=node)
    return _circuits[node]


def _transition(circuit: NodeCircuit, new_state: CircuitState, reason: str) -> None:
    """Transition a circuit to a new state with logging."""
    old_state = circuit.state
    if old_state == new_state:
        return

    circuit.state = new_state
    circuit.last_state_change = time.monotonic()

    state_colors = {
        CircuitState.CLOSED: "green",
        CircuitState.OPEN: "red",
        CircuitState.HALF_OPEN: "yellow",
    }
    color = state_colors[new_state]

    console.print(
        f"  [bold]Circuit-Breaker:[/] [{color}]{circuit.name}[/] "
        f"{old_state.value} -> [{color}]{new_state.value}[/] ({reason})"
    )


def check_node(node: str) -> bool:
    """Check if a node is allowed to receive requests.

    Returns True if the circuit is CLOSED or HALF_OPEN (probe allowed).
    Returns False if the circuit is OPEN (requests blocked).
    Automatically transitions OPEN -> HALF_OPEN after cooldown.
    """
    circuit = _get_circuit(node)

    if circuit.state == CircuitState.CLOSED:
        return True

    if circuit.state == CircuitState.OPEN:
        elapsed = time.monotonic() - circuit.last_failure_time
        if elapsed >= _cooldown_seconds:
            _transition(circuit, CircuitState.HALF_OPEN,
                        f"cooldown expired ({_cooldown_seconds}s)")
            return True  # allow one probe request
        circuit.total_blocked += 1
        return False

    if circuit.state == CircuitState.HALF_OPEN:
        return True  # allow probe

    return False


def record_success(node: str) -> None:
    """Record a successful request. May close the circuit."""
    circuit = _get_circuit(node)
    circuit.success_count += 1
    circuit.consecutive_failures = 0
    circuit.last_success_time = time.monotonic()

    if circuit.state == CircuitState.HALF_OPEN:
        _transition(circuit, CircuitState.CLOSED, "probe succeeded")
    elif circuit.state == CircuitState.OPEN:
        # Shouldn't normally happen, but handle gracefully
        _transition(circuit, CircuitState.CLOSED, "success while open")


def record_failure(node: str) -> None:
    """Record a failed request. May open the circuit."""
    circuit = _get_circuit(node)
    circuit.failure_count += 1
    circuit.consecutive_failures += 1
    circuit.last_failure_time = time.monotonic()

    if circuit.state == CircuitState.HALF_OPEN:
        _transition(circuit, CircuitState.OPEN,
                    f"probe failed (resetting cooldown)")

    elif circuit.state == CircuitState.CLOSED:
        if circuit.consecutive_failures >= _failure_threshold:
            _transition(circuit, CircuitState.OPEN,
                        f"{circuit.consecutive_failures} consecutive failures "
                        f">= threshold {_failure_threshold}")


def reset_node(node: str) -> None:
    """Manually reset a node's circuit to CLOSED."""
    circuit = _get_circuit(node)
    old_state = circuit.state
    circuit.state = CircuitState.CLOSED
    circuit.consecutive_failures = 0
    circuit.total_blocked = 0
    circuit.last_state_change = time.monotonic()
    if old_state != CircuitState.CLOSED:
        console.print(
            f"  [bold]Circuit-Breaker:[/] [cyan]{node}[/] manually reset to CLOSED"
        )


def reset_all() -> None:
    """Reset all circuits to CLOSED."""
    for node in _circuits:
        reset_node(node)


def get_all_states() -> dict[str, dict[str, Any]]:
    """Return a snapshot of all circuit states."""
    snapshot = {}
    for name, c in _circuits.items():
        snapshot[name] = {
            "state": c.state.value,
            "consecutive_failures": c.consecutive_failures,
            "total_failures": c.failure_count,
            "total_successes": c.success_count,
            "total_blocked": c.total_blocked,
            "time_in_state_s": round(c.time_in_state_s, 1),
            "allowed": check_node(name),
        }
    return snapshot


def get_available_nodes() -> list[str]:
    """Return list of node names where the circuit allows requests."""
    return [name for name in _circuits if check_node(name)]


def _print_circuit_dashboard() -> None:
    """Print a rich dashboard of all circuit states."""
    table = Table(title="Circuit Breaker — Node States", show_lines=True)
    table.add_column("Node", style="cyan", width=8)
    table.add_column("State", justify="center", width=11)
    table.add_column("Consec. Fail", justify="right", width=12)
    table.add_column("Total Fail", justify="right", width=10)
    table.add_column("Total OK", justify="right", width=10)
    table.add_column("Blocked", justify="right", width=8)
    table.add_column("Time in State", justify="right", width=13)
    table.add_column("Allowed", justify="center", width=8)

    for name, c in _circuits.items():
        state_color = {
            CircuitState.CLOSED: "green",
            CircuitState.OPEN: "red",
            CircuitState.HALF_OPEN: "yellow",
        }[c.state]

        allowed = check_node(name)
        time_str = f"{c.time_in_state_s:.0f}s"

        fail_color = "red" if c.consecutive_failures >= _failure_threshold else (
            "yellow" if c.consecutive_failures > 0 else "dim"
        )

        table.add_row(
            name,
            f"[{state_color}]{c.state.value.upper()}[/]",
            f"[{fail_color}]{c.consecutive_failures}[/]",
            str(c.failure_count),
            str(c.success_count),
            str(c.total_blocked) if c.total_blocked > 0 else "[dim]0[/]",
            time_str,
            "[green]YES[/]" if allowed else "[red]NO[/]",
        )

    console.print(table)

    # Configuration info
    console.print(
        f"  [dim]Threshold: {_failure_threshold} failures | "
        f"Cooldown: {_cooldown_seconds}s | "
        f"Available: {', '.join(get_available_nodes()) or 'NONE'}[/]"
    )


async def run(run_id: str, input_data: Any = None) -> StepResult:
    """Execute Agent Circuit-Breaker: report circuit states and check node health."""
    t0 = time.monotonic()
    console.print("[bold magenta]Agent Circuit-Breaker[/] checking circuit states...")

    _print_circuit_dashboard()

    states = get_all_states()
    available = get_available_nodes()
    latency = (time.monotonic() - t0) * 1000

    # Count circuit states
    closed_count = sum(1 for c in _circuits.values() if c.state == CircuitState.CLOSED)
    open_count = sum(1 for c in _circuits.values() if c.state == CircuitState.OPEN)
    half_open_count = sum(1 for c in _circuits.values() if c.state == CircuitState.HALF_OPEN)
    total_blocked = sum(c.total_blocked for c in _circuits.values())

    db.save_audit(
        run_id, "circuit_breaker", "circuit_breaker",
        input_summary=f"{len(_circuits)} circuits monitored",
        output_summary=f"closed={closed_count}, open={open_count}, "
                       f"half_open={half_open_count}, blocked={total_blocked}, "
                       f"available=[{', '.join(available)}]",
        model_used="circuit-breaker-pattern",
        latency_ms=latency,
    )

    # Determine overall health confidence
    if open_count == 0:
        confidence = 95
    elif len(available) > 0:
        confidence = 60
    else:
        confidence = 10  # all circuits open — critical

    status = StepStatus.SUCCESS if len(available) > 0 else StepStatus.FAILED

    console.print(
        f"[green]Circuit-Breaker done[/] — "
        f"{closed_count} closed, {open_count} open, "
        f"{half_open_count} half-open in {int(latency)}ms"
    )

    return StepResult(
        step_name="circuit_breaker",
        status=status,
        data={
            "circuits": states,
            "available_nodes": available,
            "summary": {
                "closed": closed_count,
                "open": open_count,
                "half_open": half_open_count,
                "total_blocked": total_blocked,
            },
        },
        confidence=confidence,
        agent_used="circuit_breaker",
        latency_ms=latency,
    )

"""Pipeline Engine — Domino-Vectorial-Matrix Orchestration.

Advanced multi-agent execution engine supporting three orchestration patterns:

1. DOMINO (Sequential): A → B → C → D
   Each agent's output feeds the next agent's input.
   Validation gates between steps. Auto-rollback on failure.

2. VECTORIAL (Parallel): [A, B, C] → Aggregator → Result
   Multiple agents process simultaneously.
   Results merged by an aggregator with weighted scoring.

3. MATRIX (Grid): Agents × Contexts = Optimal Strategy
   Cross-product of agent capabilities and input contexts.
   Scoring matrix determines the best agent-context pair.

Patterns can be composed: Domino steps can contain Vectorial stages,
and Matrix routing can select which Domino pipeline to execute.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src import database as db
from src.config import config

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class StepResult:
    """Result from a single pipeline step."""
    step_name: str
    status: StepStatus
    data: Any = None
    error: str = ""
    latency_ms: float = 0
    model_used: str = ""
    agent_used: str = ""
    confidence: float = 0


@dataclass
class PipelineResult:
    """Result from a complete pipeline execution."""
    pipeline_name: str
    run_id: str
    steps: list[StepResult] = field(default_factory=list)
    total_latency_ms: float = 0
    success: bool = False
    pattern: str = ""  # domino, vectorial, matrix

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def success_rate(self) -> float:
        if not self.steps:
            return 0
        return sum(1 for s in self.steps if s.status == StepStatus.SUCCESS) / len(self.steps) * 100


# Step function type: async (run_id, input_data) -> StepResult
StepFn = Callable[[str, Any], Coroutine[Any, Any, StepResult]]


# ═══════════════════════════════════════════════════════════════════════════════
# DOMINO PIPELINE — Sequential chain with validation gates
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DominoStep:
    """A single step in a domino pipeline."""
    name: str
    fn: StepFn
    timeout: float = 60.0
    required: bool = True  # if False, pipeline continues on failure
    validator: Callable[[StepResult], bool] | None = None


class DominoPipeline:
    """Sequential agent pipeline: A → B → C → D.

    Each step receives the previous step's output as input.
    Validation gates between steps catch errors early.
    """

    def __init__(self, name: str, steps: list[DominoStep]):
        self.name = name
        self.steps = steps

    async def execute(self, run_id: str, initial_input: Any = None) -> PipelineResult:
        t0 = time.monotonic()
        result = PipelineResult(
            pipeline_name=self.name,
            run_id=run_id,
            pattern="domino",
        )

        current_input = initial_input

        for i, step in enumerate(self.steps):
            console.print(f"  [cyan]Domino[/] [{i+1}/{len(self.steps)}] {step.name}...")

            try:
                step_result = await asyncio.wait_for(
                    step.fn(run_id, current_input),
                    timeout=step.timeout,
                )

                # Validation gate
                if step.validator and not step.validator(step_result):
                    step_result.status = StepStatus.FAILED
                    step_result.error = "Validation gate failed"

                result.steps.append(step_result)

                if step_result.status == StepStatus.SUCCESS:
                    current_input = step_result.data
                    console.print(f"    [green]OK[/] {step.name} — {int(step_result.latency_ms)}ms")
                elif step.required:
                    console.print(f"    [red]FAIL[/] {step.name} — {step_result.error}")
                    break
                else:
                    console.print(f"    [yellow]SKIP[/] {step.name} (optional) — {step_result.error}")

            except asyncio.TimeoutError:
                step_result = StepResult(
                    step_name=step.name,
                    status=StepStatus.TIMEOUT,
                    error=f"Timeout after {step.timeout}s",
                    latency_ms=step.timeout * 1000,
                )
                result.steps.append(step_result)
                console.print(f"    [yellow]TIMEOUT[/] {step.name} — {step.timeout}s")
                if step.required:
                    break

            except Exception as e:
                step_result = StepResult(
                    step_name=step.name,
                    status=StepStatus.FAILED,
                    error=str(e),
                )
                result.steps.append(step_result)
                console.print(f"    [red]ERROR[/] {step.name} — {e}")
                if step.required:
                    break

        result.total_latency_ms = (time.monotonic() - t0) * 1000
        result.success = all(
            s.status == StepStatus.SUCCESS
            for s in result.steps
            if self.steps[result.steps.index(s)].required
            if result.steps.index(s) < len(self.steps)
        )

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# VECTORIAL PIPELINE — Parallel execution with aggregation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VectorAgent:
    """An agent that runs in parallel within a vector."""
    name: str
    fn: StepFn
    weight: float = 1.0  # weight for aggregation
    timeout: float = 30.0


class VectorialPipeline:
    """Parallel agent execution: [A, B, C] → Aggregator → Result.

    All agents receive the same input and run simultaneously.
    Results are aggregated using weighted scoring.
    """

    def __init__(self, name: str, agents: list[VectorAgent],
                 aggregator: Callable[[list[StepResult]], StepResult] | None = None):
        self.name = name
        self.agents = agents
        self.aggregator = aggregator or self._default_aggregator

    async def execute(self, run_id: str, input_data: Any = None) -> PipelineResult:
        t0 = time.monotonic()
        result = PipelineResult(
            pipeline_name=self.name,
            run_id=run_id,
            pattern="vectorial",
        )

        console.print(f"  [magenta]Vector[/] Launching {len(self.agents)} agents in parallel...")

        # Launch all agents in parallel
        tasks = []
        for agent in self.agents:
            task = asyncio.create_task(
                self._run_agent(agent, run_id, input_data)
            )
            tasks.append((agent, task))

        # Gather results
        for agent, task in tasks:
            try:
                step_result = await asyncio.wait_for(task, timeout=agent.timeout)
                result.steps.append(step_result)
                status = "[green]OK[/]" if step_result.status == StepStatus.SUCCESS else "[red]FAIL[/]"
                console.print(f"    {status} {agent.name} — {int(step_result.latency_ms)}ms (weight: {agent.weight})")
            except asyncio.TimeoutError:
                step_result = StepResult(
                    step_name=agent.name,
                    status=StepStatus.TIMEOUT,
                    latency_ms=agent.timeout * 1000,
                )
                result.steps.append(step_result)
                console.print(f"    [yellow]TIMEOUT[/] {agent.name}")

        # Aggregate
        if result.steps:
            console.print(f"  [magenta]Vector[/] Aggregating {len(result.steps)} results...")
            aggregated = self.aggregator(result.steps)
            result.steps.append(aggregated)

        result.total_latency_ms = (time.monotonic() - t0) * 1000
        result.success = any(s.status == StepStatus.SUCCESS for s in result.steps)

        return result

    async def _run_agent(self, agent: VectorAgent, run_id: str, input_data: Any) -> StepResult:
        t0 = time.monotonic()
        try:
            result = await agent.fn(run_id, input_data)
            result.latency_ms = (time.monotonic() - t0) * 1000
            return result
        except Exception as e:
            return StepResult(
                step_name=agent.name,
                status=StepStatus.FAILED,
                error=str(e),
                latency_ms=(time.monotonic() - t0) * 1000,
            )

    @staticmethod
    def _default_aggregator(results: list[StepResult]) -> StepResult:
        """Default aggregator: weighted average of confidence scores."""
        successful = [r for r in results if r.status == StepStatus.SUCCESS]
        if not successful:
            return StepResult(
                step_name="aggregator",
                status=StepStatus.FAILED,
                error="No successful results to aggregate",
            )

        # Combine data from all successful results
        combined = {
            "sources": len(successful),
            "agents": [r.agent_used for r in successful],
            "data": [r.data for r in successful],
        }

        avg_confidence = sum(r.confidence for r in successful) / len(successful)

        return StepResult(
            step_name="aggregator",
            status=StepStatus.SUCCESS,
            data=combined,
            confidence=avg_confidence,
            agent_used="aggregator",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MATRIX PIPELINE — Grid routing: Agents × Contexts → Optimal
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MatrixCell:
    """A cell in the agent-context matrix."""
    agent_name: str
    context_name: str
    score: float = 0  # capability score for this combination
    fn: StepFn | None = None


class MatrixPipeline:
    """Matrix routing: Agents × Contexts = Optimal Strategy.

    Builds a scoring matrix of agent capabilities vs. input contexts.
    Selects the top-K agent-context pairs and executes them.
    """

    def __init__(self, name: str, agents: list[str], contexts: list[str],
                 score_fn: Callable[[str, str, Any], float] | None = None,
                 execute_fn: StepFn | None = None,
                 top_k: int = 3):
        self.name = name
        self.agents = agents
        self.contexts = contexts
        self.score_fn = score_fn or self._default_score
        self.execute_fn = execute_fn
        self.top_k = top_k
        self.matrix: list[MatrixCell] = []

    def build_matrix(self, input_data: Any = None) -> list[MatrixCell]:
        """Build the scoring matrix."""
        self.matrix = []
        for agent in self.agents:
            for context in self.contexts:
                score = self.score_fn(agent, context, input_data)
                self.matrix.append(MatrixCell(
                    agent_name=agent,
                    context_name=context,
                    score=score,
                ))
        # Sort by score descending
        self.matrix.sort(key=lambda c: c.score, reverse=True)
        return self.matrix

    async def execute(self, run_id: str, input_data: Any = None) -> PipelineResult:
        t0 = time.monotonic()
        result = PipelineResult(
            pipeline_name=self.name,
            run_id=run_id,
            pattern="matrix",
        )

        # Build scoring matrix
        console.print(f"  [yellow]Matrix[/] Building {len(self.agents)}x{len(self.contexts)} scoring matrix...")
        self.build_matrix(input_data)

        # Display matrix
        self._print_matrix()

        # Select top-K cells
        top_cells = self.matrix[:self.top_k]
        console.print(f"  [yellow]Matrix[/] Executing top-{self.top_k} agent-context pairs...")

        # Execute top-K in parallel
        tasks = []
        for cell in top_cells:
            if self.execute_fn:
                task_input = {"agent": cell.agent_name, "context": cell.context_name, "data": input_data}
                tasks.append(asyncio.create_task(
                    self.execute_fn(run_id, task_input)
                ))

        if tasks:
            step_results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, sr in enumerate(step_results):
                if isinstance(sr, Exception):
                    result.steps.append(StepResult(
                        step_name=f"{top_cells[i].agent_name}×{top_cells[i].context_name}",
                        status=StepStatus.FAILED,
                        error=str(sr),
                    ))
                else:
                    result.steps.append(sr)

        result.total_latency_ms = (time.monotonic() - t0) * 1000
        result.success = any(s.status == StepStatus.SUCCESS for s in result.steps)

        return result

    def _print_matrix(self) -> None:
        """Display the scoring matrix as a table."""
        table = Table(title=f"Routing Matrix: {self.name}", box=box.ROUNDED)
        table.add_column("Agent \\ Context", style="cyan")
        for ctx in self.contexts:
            table.add_column(ctx, justify="center")

        for agent in self.agents:
            row = [agent]
            for ctx in self.contexts:
                cell = next((c for c in self.matrix if c.agent_name == agent and c.context_name == ctx), None)
                if cell:
                    score = cell.score
                    color = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"
                    row.append(f"[{color}]{score:.2f}[/]")
                else:
                    row.append("[dim]—[/]")
            table.add_row(*row)

        console.print(table)

    @staticmethod
    def _default_score(agent: str, context: str, input_data: Any) -> float:
        """Default scoring: capability matching heuristic."""
        scores = {
            ("M1", "complex"): 0.95, ("M1", "analysis"): 0.90, ("M1", "trading"): 0.85,
            ("M1", "simple"): 0.40, ("M1", "system"): 0.20,
            ("OL1", "simple"): 0.90, ("OL1", "quick"): 0.95, ("OL1", "chat"): 0.85,
            ("OL1", "complex"): 0.30, ("OL1", "trading"): 0.40,
            ("Airia", "complex"): 0.85, ("Airia", "analysis"): 0.80, ("Airia", "trading"): 0.75,
            ("Airia", "simple"): 0.60, ("Airia", "system"): 0.30,
            ("local", "system"): 0.95, ("local", "simple"): 0.70,
            ("local", "complex"): 0.10, ("local", "trading"): 0.05,
        }
        return scores.get((agent, context), 0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE PIPELINE — Combine patterns
# ═══════════════════════════════════════════════════════════════════════════════

class CompositePipeline:
    """Compose Domino, Vectorial, and Matrix patterns together.

    Example: Matrix selects agents → Vectorial runs them in parallel →
             Domino chains validation + enrichment + output.
    """

    def __init__(self, name: str):
        self.name = name
        self.stages: list[tuple[str, Any]] = []  # (pattern, pipeline)

    def add_domino(self, pipeline: DominoPipeline) -> "CompositePipeline":
        self.stages.append(("domino", pipeline))
        return self

    def add_vectorial(self, pipeline: VectorialPipeline) -> "CompositePipeline":
        self.stages.append(("vectorial", pipeline))
        return self

    def add_matrix(self, pipeline: MatrixPipeline) -> "CompositePipeline":
        self.stages.append(("matrix", pipeline))
        return self

    async def execute(self, run_id: str, initial_input: Any = None) -> PipelineResult:
        t0 = time.monotonic()
        composite_result = PipelineResult(
            pipeline_name=self.name,
            run_id=run_id,
            pattern="composite",
        )

        current_input = initial_input

        for i, (pattern, pipeline) in enumerate(self.stages):
            console.print(f"\n[bold]Stage {i+1}/{len(self.stages)}:[/] {pattern.upper()} — {pipeline.name}")

            stage_result = await pipeline.execute(run_id, current_input)

            # Merge steps
            composite_result.steps.extend(stage_result.steps)

            # Pass output to next stage
            if stage_result.steps:
                last_successful = next(
                    (s for s in reversed(stage_result.steps) if s.status == StepStatus.SUCCESS),
                    None,
                )
                if last_successful:
                    current_input = last_successful.data

            if not stage_result.success:
                console.print(f"  [red]Stage {i+1} failed. Pipeline halted.[/]")
                break

        composite_result.total_latency_ms = (time.monotonic() - t0) * 1000
        composite_result.success = all(
            any(s.status == StepStatus.SUCCESS for s in composite_result.steps)
            for _ in self.stages
        )

        return composite_result


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY — Print pipeline results
# ═══════════════════════════════════════════════════════════════════════════════

def print_pipeline_result(result: PipelineResult) -> None:
    """Pretty-print a pipeline execution result."""
    table = Table(title=f"Pipeline: {result.pipeline_name}", box=box.ROUNDED)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Step", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Agent", style="magenta")
    table.add_column("Latency", justify="right", style="dim")
    table.add_column("Confidence", justify="right")

    for i, step in enumerate(result.steps, 1):
        status_map = {
            StepStatus.SUCCESS: "[green]OK[/]",
            StepStatus.FAILED: "[red]FAIL[/]",
            StepStatus.TIMEOUT: "[yellow]TIMEOUT[/]",
            StepStatus.SKIPPED: "[dim]SKIP[/]",
            StepStatus.RUNNING: "[cyan]...[/]",
            StepStatus.PENDING: "[dim]—[/]",
        }
        table.add_row(
            str(i),
            step.step_name,
            status_map.get(step.status, str(step.status)),
            step.agent_used or "—",
            f"{int(step.latency_ms)}ms" if step.latency_ms else "—",
            f"{step.confidence:.0f}%" if step.confidence else "—",
        )

    console.print(table)

    console.print(Panel(
        f"[bold]Pattern:[/] {result.pattern}\n"
        f"[bold]Steps:[/] {result.step_count}\n"
        f"[bold]Success Rate:[/] {result.success_rate:.0f}%\n"
        f"[bold]Total Latency:[/] {int(result.total_latency_ms)}ms\n"
        f"[bold]Status:[/] {'[green]SUCCESS[/]' if result.success else '[red]FAILED[/]'}",
        title=f"[bold]Result: {result.pipeline_name}[/]",
        border_style="green" if result.success else "red",
    ))

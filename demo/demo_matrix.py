"""Demo Matrix — Pipeline Domino-Vectorial-Matrix Orchestration (< 3 min).

Demonstrates the three orchestration patterns working together:
1. Matrix routing selects optimal agents
2. Vectorial parallel execution across providers
3. Domino sequential chain with validation gates

Usage:
    uv run python main.py demo-matrix
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.pipeline_engine import (
    StepResult, StepStatus, PipelineResult,
    DominoPipeline, DominoStep,
    VectorialPipeline, VectorAgent,
    MatrixPipeline,
    CompositePipeline,
    print_pipeline_result,
)

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def narrate(text: str, pause: float = 0.8) -> None:
    console.print(f"\n[bold white on blue]  {text}  [/]")
    time.sleep(pause)


def section_divider(title: str) -> None:
    console.print()
    console.rule(f"[bold yellow]{title}[/]", style="yellow")
    console.print()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP FUNCTIONS — Real agent calls for the pipeline
# ═══════════════════════════════════════════════════════════════════════════════

async def step_scan_markets(run_id: str, input_data: Any) -> StepResult:
    """Domino Step 1: Scan market data."""
    t0 = time.monotonic()
    from src.services.market_data import fetch_all_market_data
    signals = await fetch_all_market_data()
    return StepResult(
        step_name="market_scan",
        status=StepStatus.SUCCESS,
        data=signals,
        confidence=90,
        agent_used="market_scanner",
        latency_ms=(time.monotonic() - t0) * 1000,
    )


async def step_validate_signals(run_id: str, input_data: Any) -> StepResult:
    """Domino Step 2: Validate signal quality."""
    from src.agents.matrix.validator import run as validate_run, SIGNAL_RULES
    return await validate_run(run_id, input_data, SIGNAL_RULES, "signal_validation")


async def step_enrich_metadata(run_id: str, input_data: Any) -> StepResult:
    """Domino Step 3: Enrich with metadata."""
    from src.agents.matrix.enricher import enrich_with_metadata
    return await enrich_with_metadata(run_id, input_data)


async def step_optimize_routing(run_id: str, input_data: Any) -> StepResult:
    """Domino Step 4: Optimize routing for analysis."""
    from src.agents.matrix.optimizer import run as optimizer_run
    return await optimizer_run(run_id, input_data)


async def step_analyze_m1(run_id: str, input_data: Any) -> StepResult:
    """Vectorial: Analyze with M1 (qwen3-30b)."""
    t0 = time.monotonic()
    from src.services.lm_cluster import query_lm
    prompt = f"Analyse ces signaux de marche et identifie les risques: {str(input_data)[:1500]}"
    result = await query_lm(prompt, "M1", system="Tu es un analyste financier expert.")
    latency = (time.monotonic() - t0) * 1000
    if result.get("ok"):
        return StepResult(
            step_name="analysis_M1",
            status=StepStatus.SUCCESS,
            data=result["content"],
            confidence=85,
            agent_used="M1",
            model_used=result.get("model", "qwen3-30b"),
            latency_ms=latency,
        )
    return StepResult("analysis_M1", StepStatus.FAILED, error=result.get("error", "M1 failed"), latency_ms=latency)


async def step_analyze_ol1(run_id: str, input_data: Any) -> StepResult:
    """Vectorial: Quick analysis with OL1 (qwen3:1.7b)."""
    t0 = time.monotonic()
    from src.services.lm_cluster import query_ollama
    prompt = f"Resume les risques principaux: {str(input_data)[:800]}"
    result = await query_ollama(prompt, "OL1", system="Resume en 3 points.")
    latency = (time.monotonic() - t0) * 1000
    if result.get("ok"):
        return StepResult(
            step_name="analysis_OL1",
            status=StepStatus.SUCCESS,
            data=result["content"],
            confidence=70,
            agent_used="OL1",
            model_used=result.get("model", "qwen3:1.7b"),
            latency_ms=latency,
        )
    return StepResult("analysis_OL1", StepStatus.FAILED, error=result.get("error", "OL1 failed"), latency_ms=latency)


async def step_analyze_airia(run_id: str, input_data: Any) -> StepResult:
    """Vectorial: Analyze with Airia cloud."""
    t0 = time.monotonic()
    from src.airia_bridge import bridge
    if not bridge.is_available:
        return StepResult("analysis_Airia", StepStatus.SKIPPED, error="Airia not configured",
                          latency_ms=(time.monotonic() - t0) * 1000)
    try:
        result = await bridge.query(
            "market_intelligence",
            f"Analyse de risque: {str(input_data)[:1000]}",
        )
        return StepResult(
            step_name="analysis_Airia",
            status=StepStatus.SUCCESS,
            data=str(result)[:500],
            confidence=80,
            agent_used="Airia",
            model_used="gpt-5.1",
            latency_ms=(time.monotonic() - t0) * 1000,
        )
    except Exception as e:
        return StepResult("analysis_Airia", StepStatus.FAILED, error=str(e),
                          latency_ms=(time.monotonic() - t0) * 1000)


async def step_matrix_execute(run_id: str, input_data: Any) -> StepResult:
    """Matrix cell execution."""
    t0 = time.monotonic()
    agent = input_data.get("agent", "local") if isinstance(input_data, dict) else "local"
    context = input_data.get("context", "simple") if isinstance(input_data, dict) else "simple"

    return StepResult(
        step_name=f"{agent}×{context}",
        status=StepStatus.SUCCESS,
        data={"agent": agent, "context": context, "optimized": True},
        confidence=75,
        agent_used=agent,
        latency_ms=(time.monotonic() - t0) * 1000,
    )


async def step_aggregate(run_id: str, input_data: Any) -> StepResult:
    """Aggregate vectorial results."""
    from src.agents.matrix.aggregator import run as agg_run, CONSENSUS_STRATEGY
    if isinstance(input_data, list):
        return await agg_run(run_id, input_data, CONSENSUS_STRATEGY)
    return StepResult("aggregation", StepStatus.SUCCESS, data=input_data, confidence=80, agent_used="aggregator")


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO STEPS
# ═══════════════════════════════════════════════════════════════════════════════

async def demo_intro(clear_screen: bool = True) -> None:
    """Cinematic intro."""
    if clear_screen:
        console.clear()

    console.print(Panel(
        Align.center(Text.from_markup(
            "[bold yellow]"
            " __  __    _  _____ ____  _____  __\n"
            "|  \\/  |  / \\|_   _|  _ \\|_ _\\ \\/ /\n"
            "| |\\/| | / _ \\ | | | |_) || | \\  /\n"
            "| |  | |/ ___ \\| | |  _ < | | /  \\\n"
            "|_|  |_/_/   \\_\\_| |_| \\_\\___/_/\\_\\\n"
            "[/]\n\n"
            "[bold white]Pipeline Domino · Vectorial · Matrix Orchestration[/]\n"
            "[dim]Airia Sentinel Platform — Advanced Agent Architecture[/]"
        )),
        border_style="bright_yellow",
        padding=(1, 2),
    ))

    await asyncio.sleep(0.5)

    console.print(Panel(
        "[bold white]THREE ORCHESTRATION PATTERNS[/]\n\n"
        "  [yellow]1. DOMINO[/]      A → B → C → D\n"
        "                  Sequential chain with validation gates\n"
        "                  Each step feeds the next. Auto-rollback on failure.\n\n"
        "  [magenta]2. VECTORIAL[/]   [A, B, C] → Aggregator → Result\n"
        "                  Parallel execution across M1 + OL1 + Airia\n"
        "                  Weighted consensus with dissent detection.\n\n"
        "  [cyan]3. MATRIX[/]      Agents × Contexts = Optimal Strategy\n"
        "                  Scoring grid selects best agent-context pairs\n"
        "                  Top-K parallel execution.\n\n"
        "[dim]Patterns compose: Matrix → Vectorial → Domino in a single pipeline[/]",
        title="[bold yellow]Architecture[/]",
        border_style="yellow",
        padding=(0, 2),
    ))

    await asyncio.sleep(1)


async def demo_domino() -> PipelineResult:
    """Demo 1: Domino pipeline."""
    section_divider("PATTERN 1: DOMINO PIPELINE")

    narrate("Sequential chain: Scan → Validate → Enrich → Optimize", pause=1)

    pipeline = DominoPipeline("Domino-Risk-Analysis", [
        DominoStep("market_scan", step_scan_markets, timeout=30),
        DominoStep("validation", step_validate_signals, timeout=5),
        DominoStep("enrichment", step_enrich_metadata, timeout=5),
        DominoStep("optimization", step_optimize_routing, timeout=10),
    ])

    from src import database as db
    db.init_db()

    import uuid
    run_id = f"matrix-{uuid.uuid4().hex[:8]}"

    result = await pipeline.execute(run_id)
    print_pipeline_result(result)

    await asyncio.sleep(0.5)
    return result


async def demo_vectorial(domino_data: Any) -> PipelineResult:
    """Demo 2: Vectorial parallel execution."""
    section_divider("PATTERN 2: VECTORIAL PARALLEL EXECUTION")

    narrate("3 AI providers analyze simultaneously: M1 + OL1 + Airia", pause=1)

    pipeline = VectorialPipeline("Vector-3Way-Analysis", [
        VectorAgent("M1_Deep", step_analyze_m1, weight=0.45, timeout=15),
        VectorAgent("OL1_Quick", step_analyze_ol1, weight=0.20, timeout=10),
        VectorAgent("Airia_Cloud", step_analyze_airia, weight=0.35, timeout=15),
    ])

    import uuid
    run_id = f"vec-{uuid.uuid4().hex[:8]}"

    result = await pipeline.execute(run_id, domino_data)
    print_pipeline_result(result)

    await asyncio.sleep(0.5)
    return result


async def demo_matrix() -> PipelineResult:
    """Demo 3: Matrix routing."""
    section_divider("PATTERN 3: MATRIX ROUTING")

    narrate("Building Agent × Context scoring matrix...", pause=1)

    pipeline = MatrixPipeline(
        "Matrix-Routing",
        agents=["M1", "OL1", "Airia", "local"],
        contexts=["complex", "simple", "trading", "system", "analysis"],
        execute_fn=step_matrix_execute,
        top_k=3,
    )

    import uuid
    run_id = f"mtx-{uuid.uuid4().hex[:8]}"

    result = await pipeline.execute(run_id)
    print_pipeline_result(result)

    await asyncio.sleep(0.5)
    return result


async def demo_composite() -> PipelineResult:
    """Demo 4: All patterns composed together."""
    section_divider("COMPOSITE: DOMINO + VECTORIAL + MATRIX")

    narrate("Full composite pipeline: all 3 patterns in sequence...", pause=1)

    composite = CompositePipeline("Sentinel-Matrix-Full")

    # Stage 1: Domino (scan + validate + enrich)
    composite.add_domino(DominoPipeline("Data-Acquisition", [
        DominoStep("scan", step_scan_markets, timeout=30),
        DominoStep("validate", step_validate_signals, timeout=5),
        DominoStep("enrich", step_enrich_metadata, timeout=5),
    ]))

    # Stage 2: Vectorial (parallel analysis)
    composite.add_vectorial(VectorialPipeline("Multi-AI-Analysis", [
        VectorAgent("M1", step_analyze_m1, weight=0.45, timeout=15),
        VectorAgent("OL1", step_analyze_ol1, weight=0.20, timeout=10),
    ]))

    # Stage 3: Matrix (optimal routing)
    composite.add_matrix(MatrixPipeline(
        "Final-Routing",
        agents=["M1", "OL1", "Airia", "local"],
        contexts=["complex", "simple", "trading"],
        execute_fn=step_matrix_execute,
        top_k=2,
    ))

    import uuid
    run_id = f"comp-{uuid.uuid4().hex[:8]}"
    from src import database as db
    db.init_db()

    result = await composite.execute(run_id)
    print_pipeline_result(result)

    await asyncio.sleep(0.5)
    return result


async def demo_summary(results: list[PipelineResult]) -> None:
    """Final summary."""
    section_divider("DEMO COMPLETE")

    table = Table(title="Pipeline Execution Summary", box=box.ROUNDED)
    table.add_column("Pipeline", style="cyan")
    table.add_column("Pattern", style="magenta")
    table.add_column("Steps", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Status", justify="center")

    for r in results:
        status = "[green]OK[/]" if r.success else "[red]FAIL[/]"
        table.add_row(
            r.pipeline_name,
            r.pattern.upper(),
            str(r.step_count),
            f"{r.success_rate:.0f}%",
            f"{int(r.total_latency_ms)}ms",
            status,
        )

    console.print(table)

    total_steps = sum(r.step_count for r in results)
    total_latency = sum(r.total_latency_ms for r in results)
    overall_success = sum(1 for r in results if r.success) / len(results) * 100

    console.print(Panel(
        f"[bold]Total Pipelines:[/] {len(results)}\n"
        f"[bold]Total Steps:[/] {total_steps}\n"
        f"[bold]Overall Success:[/] {overall_success:.0f}%\n"
        f"[bold]Total Latency:[/] {int(total_latency)}ms\n"
        f"[bold]Patterns Used:[/] Domino + Vectorial + Matrix + Composite",
        title="[bold green]Matrix Architecture Results[/]",
        border_style="green",
    ))

    await asyncio.sleep(0.5)

    console.print(Panel(
        "[bold white]Matrix Architecture Stack[/]\n\n"
        "  [yellow]Domino[/]        Sequential chain with validation gates\n"
        "  [magenta]Vectorial[/]     Parallel multi-AI with weighted aggregation\n"
        "  [cyan]Matrix[/]        Agent×Context scoring grid\n"
        "  [green]Composite[/]     All patterns composed together\n\n"
        "  [bold]Agents:[/]       Validator, Aggregator, Optimizer, Enricher\n"
        "  [bold]Providers:[/]    M1 (qwen3-30b) + OL1 (qwen3:1.7b) + Airia (cloud)\n"
        "  [bold]Audit:[/]        Full traceability in SQLite\n\n"
        "[dim italic]From simple chains to complex multi-dimensional orchestration.[/]",
        title="[bold yellow]Stack[/]",
        border_style="yellow",
        padding=(0, 2),
    ))

    console.print()
    console.print(Align.center(Text.from_markup(
        "[bold bright_yellow]MATRIX[/]  [dim]|[/]  "
        "[bold white]Domino-Vectorial-Matrix Orchestration[/]  [dim]|[/]  "
        "[bold green]Advanced Architecture[/]"
    )))
    console.print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

async def run_demo(clear_screen: bool = True) -> None:
    """Run the Matrix demo showcasing all 3 orchestration patterns."""
    try:
        results = []

        await demo_intro(clear_screen=clear_screen)

        # Pattern 1: Domino
        r1 = await demo_domino()
        results.append(r1)

        # Pattern 2: Vectorial (using domino output)
        domino_data = None
        if r1.steps:
            last_ok = next((s for s in reversed(r1.steps) if s.status == StepStatus.SUCCESS), None)
            if last_ok:
                domino_data = last_ok.data
        r2 = await demo_vectorial(domino_data)
        results.append(r2)

        # Pattern 3: Matrix
        r3 = await demo_matrix()
        results.append(r3)

        # Pattern 4: Composite (all together)
        r4 = await demo_composite()
        results.append(r4)

        # Summary
        await demo_summary(results)

    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted.[/]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/]")
        import traceback
        traceback.print_exc()

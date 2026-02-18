"""Demo JARVIS — Multi-Agent Personal AI Assistant (< 3 min).

Demonstrates real-time intent classification, multi-AI routing,
fallback chains, and execution across different sub-agents.

Usage:
    uv run python main.py demo-jarvis
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

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def narrate(text: str, pause: float = 0.8) -> None:
    console.print(f"\n[bold white on blue]  {text}  [/]")
    time.sleep(pause)


def section_divider(title: str) -> None:
    console.print()
    console.rule(f"[bold cyan]{title}[/]", style="cyan")
    console.print()


def typewriter(text: str, delay: float = 0.02) -> None:
    for char in text:
        console.print(char, end="", highlight=False)
        time.sleep(delay)
    console.print()


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO COMMANDS — realistic user inputs
# ═══════════════════════════════════════════════════════════════════════════════

DEMO_COMMANDS = [
    {
        "input": "status du systeme",
        "description": "System status check — local execution, no AI needed",
        "expected_domain": "system",
        "expected_agent": "ia-system",
    },
    {
        "input": "ouvre chrome",
        "description": "Application launch — direct system command",
        "expected_domain": "system",
        "expected_agent": "ia-system",
    },
    {
        "input": "lance le pipeline sentinel complet",
        "description": "Pipeline trigger — routes to orchestrator (priority over system)",
        "expected_domain": "pipeline",
        "expected_agent": "orchestrator",
    },
    {
        "input": "cherche sur google les nouvelles du jour",
        "description": "Web search — routes to ia-system for browser control",
        "expected_domain": "web",
        "expected_agent": "ia-system",
    },
    {
        "input": "explique comment fonctionne une blockchain",
        "description": "Deep analysis — routes to ia-deep (qwen3-30b on M1)",
        "expected_domain": "analysis",
        "expected_agent": "ia-deep",
    },
    {
        "input": "scan crypto btc maintenant",
        "description": "Trading scan — routes to ia-trading (M1 with market data)",
        "expected_domain": "trading",
        "expected_agent": "ia-trading",
    },
    {
        "input": "bonjour, quel temps fait-il aujourd'hui?",
        "description": "Casual question — routes to ia-fast (qwen3:1.7b, quick answer)",
        "expected_domain": "conversation",
        "expected_agent": "ia-fast",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO STEPS
# ═══════════════════════════════════════════════════════════════════════════════

async def step_intro(clear_screen: bool = True) -> None:
    """Intro: present the JARVIS use case."""
    if clear_screen:
        console.clear()

    console.print(Panel(
        Align.center(Text.from_markup(
            "[bold cyan]"
            "      _    _    ______     _____ ____\n"
            "     | |  / \\  |  _ \\ \\   / /_ _/ ___|\n"
            "  _  | | / _ \\ | |_) \\ \\ / / | |\\___ \\\n"
            " | |_| |/ ___ \\|  _ < \\ V /  | | ___) |\n"
            "  \\___/_/   \\_\\_| \\_\\ \\_/  |___|____/\n"
            "[/]\n\n"
            "[bold white]Personal AI Assistant — Multi-Agent Multi-Model[/]\n"
            "[dim]Airia Sentinel Platform — Use Case #3[/]"
        )),
        border_style="bright_cyan",
        padding=(1, 2),
    ))

    await asyncio.sleep(0.5)

    console.print(Panel(
        "[bold red]THE PROBLEM[/]\n\n"
        "Users interact with AI through a single interface, but:\n"
        "  [dim]>[/] Simple tasks waste expensive compute on powerful models\n"
        "  [dim]>[/] Complex tasks fail on lightweight models\n"
        "  [dim]>[/] System commands shouldn't need AI at all\n"
        "  [dim]>[/] No fallback when a model is unavailable\n\n"
        "[italic]One-size-fits-all AI wastes resources and degrades experience.[/]",
        title="[bold red]Problem[/]",
        border_style="red",
        padding=(0, 2),
    ))

    await asyncio.sleep(1)

    console.print(Panel(
        "[bold green]THE SOLUTION: JARVIS MULTI-AGENT ROUTER[/]\n\n"
        "2 agents working together to route ANY command optimally:\n\n"
        "  [cyan]Agent 1[/] [bold]Intent Classifier[/]   Rule-based parsing (< 1ms)\n"
        "                                7 domains, 12 actions, entity extraction\n\n"
        "  [cyan]Agent 2[/] [bold]Execution Engine[/]     Routes to the right sub-agent:\n"
        "                                ia-deep (qwen3-30b) for complex tasks\n"
        "                                ia-fast (qwen3:1.7b) for quick answers\n"
        "                                ia-system for local operations\n"
        "                                ia-trading for market analysis\n"
        "                                orchestrator for pipeline execution\n\n"
        "[dim]Automatic timeout handling + fallback chains + audit trail[/]",
        title="[bold green]Solution[/]",
        border_style="green",
        padding=(0, 2),
    ))

    await asyncio.sleep(1)


async def step_architecture() -> None:
    """Show the JARVIS architecture."""
    section_divider("ARCHITECTURE")

    console.print(Panel(
        "[bold white]JARVIS Pipeline Flow:[/]\n\n"
        "  User Input\n"
        "      |\n"
        "      v\n"
        "  +---------------------+\n"
        "  | [bold]Intent Classifier[/]   |  Rule-based (< 1ms)\n"
        "  | Domain + Action      |  + Entity extraction\n"
        "  | + Confidence score   |  (apps, URLs, pairs)\n"
        "  +---------------------+\n"
        "      |\n"
        "      v\n"
        "  +---------------------+\n"
        "  | [bold]Execution Engine[/]    |  Routing table:\n"
        "  |                     |  ia-deep  -> M1 (90s timeout)\n"
        "  | Timeout handling    |  ia-fast  -> OL1 (15s timeout)\n"
        "  | Fallback chains     |  ia-system -> local (10s)\n"
        "  | Audit logging       |  ia-trading -> M1 (60s)\n"
        "  +---------------------+  orchestrator -> pipeline\n"
        "      |\n"
        "      v\n"
        "  Response to User\n\n"
        "  [dim]Fallback: ia-deep -> ia-fast -> airia[/]\n"
        "  [dim]          ia-trading -> ia-deep -> ia-fast[/]",
        title="[bold cyan]Multi-Agent Architecture[/]",
        border_style="cyan",
        padding=(0, 2),
    ))

    await asyncio.sleep(1)


async def step_system_check() -> None:
    """Check system readiness for JARVIS demo."""
    section_divider("SYSTEM CHECK")

    narrate("Checking AI infrastructure...", pause=0.5)

    from src.services.lm_cluster import cluster_health

    health = await cluster_health()

    table = Table(title="JARVIS Infrastructure", box=box.ROUNDED)
    table.add_column("Sub-Agent", style="cyan")
    table.add_column("Backend", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    # Check each backend
    m1_online = any(n.get("online") and n["node"] == "M1" for n in health["nodes"])
    ol1_online = any(n.get("online") and n["node"] == "OL1" for n in health["nodes"])

    table.add_row(
        "ia-deep", "LM Studio M1",
        "[bold green]ONLINE[/]" if m1_online else "[red]OFFLINE[/]",
        "qwen3-30b, 43GB VRAM" if m1_online else "Fallback to ia-fast",
    )
    table.add_row(
        "ia-fast", "Ollama OL1",
        "[bold green]ONLINE[/]" if ol1_online else "[red]OFFLINE[/]",
        "qwen3:1.7b, quick responses" if ol1_online else "Fallback to ia-deep",
    )
    table.add_row(
        "ia-system", "Local",
        "[bold green]ONLINE[/]",
        "Direct execution, no AI",
    )
    table.add_row(
        "ia-trading", "LM Studio M1",
        "[bold green]ONLINE[/]" if m1_online else "[red]OFFLINE[/]",
        "Market analysis + CCXT" if m1_online else "Fallback chain active",
    )
    table.add_row(
        "orchestrator", "Pipeline",
        "[bold green]READY[/]",
        "Sentinel 4-agent pipeline",
    )

    console.print(table)
    await asyncio.sleep(0.5)


async def step_run_commands() -> list[dict]:
    """Execute the demo commands through JARVIS pipeline."""
    section_divider("LIVE DEMO: PROCESSING COMMANDS")

    from src.agents.jarvis.intent_classifier import classify, ParsedIntent
    from src.agents.jarvis.execution_engine import execute, AGENT_ROUTES, FALLBACK_CHAIN
    from src import database as db
    import uuid

    db.init_db()
    run_id = f"jarvis-demo-{uuid.uuid4().hex[:6]}"
    results = []

    for i, cmd in enumerate(DEMO_COMMANDS, 1):
        console.print(f"\n[bold white]Command {i}/{len(DEMO_COMMANDS)}:[/]")
        console.print(f"  [dim]{cmd['description']}[/]")

        # Typewriter effect for input
        console.print("  [bold cyan]>[/] ", end="")
        typewriter(f'"{cmd["input"]}"', delay=0.015)

        await asyncio.sleep(0.3)

        # Phase 1: Classify
        t0 = time.monotonic()
        intent = classify(cmd["input"])
        classify_ms = (time.monotonic() - t0) * 1000

        # Show classification
        domain_match = intent.domain.value == cmd["expected_domain"]
        agent_match = intent.suggested_agent == cmd["expected_agent"]

        intent_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        intent_table.add_column("Field", style="dim")
        intent_table.add_column("Value")

        domain_str = f"[{'green' if domain_match else 'red'}]{intent.domain.value}[/]"
        agent_str = f"[{'green' if agent_match else 'red'}]{intent.suggested_agent}[/]"

        intent_table.add_row("Domain", domain_str)
        intent_table.add_row("Action", intent.action.value)
        intent_table.add_row("Agent", agent_str)
        intent_table.add_row("Confidence", f"{intent.confidence:.0f}%")
        intent_table.add_row("Requires AI", "[yellow]Yes[/]" if intent.requires_ai else "[green]No[/]")
        intent_table.add_row("Classify time", f"[dim]{classify_ms:.2f}ms[/]")

        if intent.entities:
            intent_table.add_row("Entities", ", ".join(f"{k}={v}" for k, v in intent.entities.items()))

        console.print(intent_table)

        # Phase 2: Execute (with short timeout for demo)
        t1 = time.monotonic()
        try:
            result = await asyncio.wait_for(execute(intent), timeout=12)
            exec_ms = (time.monotonic() - t1) * 1000

            status = "[green]OK[/]" if result.success else "[red]FAIL[/]"
            console.print(
                f"  {status} Agent: [bold]{result.agent_used}[/] "
                f"({result.model_used or 'local'}) — {int(exec_ms)}ms"
                f"{' [yellow]FALLBACK[/]' if result.fallback_used else ''}"
            )

            if result.response:
                response_preview = result.response[:120].replace("\n", " ")
                console.print(f"  [dim]Response: {response_preview}...[/]")

        except asyncio.TimeoutError:
            exec_ms = (time.monotonic() - t1) * 1000
            console.print(f"  [yellow]TIMEOUT[/] — {int(exec_ms)}ms (demo limit 12s)")
            result = None

        except Exception as e:
            exec_ms = (time.monotonic() - t1) * 1000
            console.print(f"  [red]ERROR[/] — {e}")
            result = None

        results.append({
            "input": cmd["input"],
            "domain": intent.domain.value,
            "action": intent.action.value,
            "agent": intent.suggested_agent,
            "confidence": intent.confidence,
            "classify_ms": classify_ms,
            "exec_ms": exec_ms,
            "success": result.success if result else False,
            "fallback": result.fallback_used if result else False,
        })

        # Save audit
        db.save_audit(
            run_id, "jarvis_demo", "demo",
            input_summary=f"'{cmd['input'][:60]}'",
            output_summary=f"{intent.domain.value}/{intent.action.value} -> {intent.suggested_agent}",
            model_used=result.model_used if result else "timeout",
            latency_ms=classify_ms + exec_ms,
        )

        await asyncio.sleep(0.5)

    return results


async def step_summary(results: list[dict]) -> None:
    """Final summary."""
    section_divider("DEMO COMPLETE")

    # Results table
    table = Table(title="JARVIS Demo Results", box=box.ROUNDED)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Command", max_width=35)
    table.add_column("Domain", style="cyan")
    table.add_column("Agent", style="magenta")
    table.add_column("Conf.", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Time", justify="right", style="dim")

    for i, r in enumerate(results, 1):
        status = "[green]OK[/]" if r["success"] else "[red]FAIL[/]"
        if r["fallback"]:
            status += " [yellow]FB[/]"
        table.add_row(
            str(i),
            r["input"][:35],
            r["domain"],
            r["agent"],
            f"{r['confidence']:.0f}%",
            status,
            f"{r['exec_ms']:.0f}ms",
        )

    console.print(table)

    await asyncio.sleep(0.5)

    # Key metrics
    total_commands = len(results)
    success_rate = sum(1 for r in results if r["success"]) / total_commands * 100 if total_commands else 0
    avg_classify = sum(r["classify_ms"] for r in results) / total_commands if total_commands else 0
    avg_exec = sum(r["exec_ms"] for r in results) / total_commands if total_commands else 0
    fallbacks = sum(1 for r in results if r["fallback"])
    domains_used = len(set(r["domain"] for r in results))
    agents_used = len(set(r["agent"] for r in results))

    metrics = Table(box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 2))
    metrics.add_column("Metric", style="cyan")
    metrics.add_column("Value", style="bold white", justify="right")

    metrics.add_row("Commands processed", str(total_commands))
    metrics.add_row("Success rate", f"[green]{success_rate:.0f}%[/]")
    metrics.add_row("Avg classification time", f"{avg_classify:.2f}ms")
    metrics.add_row("Avg execution time", f"{avg_exec:.0f}ms")
    metrics.add_row("Domains covered", f"{domains_used}/7")
    metrics.add_row("Agents used", f"{agents_used}")
    metrics.add_row("Fallbacks triggered", f"[yellow]{fallbacks}[/]")

    console.print(Panel(
        metrics,
        title="[bold green]JARVIS Performance[/]",
        border_style="green",
        padding=(0, 1),
    ))

    await asyncio.sleep(0.5)

    # Tech stack
    console.print(Panel(
        "[bold white]JARVIS Technology Stack[/]\n\n"
        "  [cyan]Intent Classifier[/]   Rule-based parsing (< 1ms, 7 domains, 12 actions)\n"
        "  [cyan]Entity Extraction[/]   Apps, URLs, file paths, trading pairs\n"
        "  [cyan]Execution Engine[/]    5 sub-agents with timeout + fallback chains\n"
        "  [cyan]ia-deep[/]             LM Studio M1 — qwen3-30b (43GB VRAM, 5 GPU)\n"
        "  [cyan]ia-fast[/]             Ollama OL1 — qwen3:1.7b (lightweight)\n"
        "  [cyan]ia-system[/]           Local execution (no AI overhead)\n"
        "  [cyan]ia-trading[/]          Market analysis with CCXT integration\n"
        "  [cyan]Audit[/]               Full traceability in SQLite\n\n"
        "[dim italic]The right AI model for the right task — automatically.[/]",
        title="[bold cyan]Stack[/]",
        border_style="cyan",
        padding=(0, 2),
    ))

    await asyncio.sleep(1)

    console.print()
    console.print(Align.center(Text.from_markup(
        "[bold bright_cyan]JARVIS[/]  [dim]|[/]  "
        "[bold white]Multi-Agent Personal AI Assistant[/]  [dim]|[/]  "
        "[bold green]Use Case #3[/]"
    )))
    console.print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

async def run_demo(clear_screen: bool = True) -> None:
    """Run the JARVIS demo with real command processing."""
    try:
        await step_intro(clear_screen=clear_screen)
        await step_architecture()
        await step_system_check()
        results = await step_run_commands()
        await step_summary(results)

    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted.[/]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/]")
        import traceback
        traceback.print_exc()

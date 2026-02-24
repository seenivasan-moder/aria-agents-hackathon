#!/usr/bin/env python3
"""Airia Sentinel — CLI entry point.

Usage:
    python main.py                  # Full pipeline v2 (10 agents, default)
    python main.py pipeline-v1      # Original 4-agent pipeline
    python main.py dashboard        # Start HITL dashboard server
    python main.py scan             # Market scan only
    python main.py status           # Cluster health check
    python main.py demo             # Demo scenario for video
    python main.py hitl             # Start HITL webhook server
    python main.py meta [prompt]    # Meta-Exchange: multi-AI orchestration
    python main.py organize [dir]   # Organizer: file scan + dedup
    python main.py jarvis [cmd]     # JARVIS: personal AI assistant
    python main.py demo-org [dir]   # Demo: Organizer (real file scan)
    python main.py demo-jarvis      # Demo: JARVIS (multi-agent routing)
    python main.py demo-matrix      # Demo: Matrix pipeline orchestration
    python main.py demo-all         # Presentation: 3 use cases (Sentinel+Organizer+JARVIS)
"""

from __future__ import annotations

import asyncio
import os
import sys

# Fix Windows cp1252 encoding issues with unicode (Airia returns French text)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console

console = Console(force_terminal=True)

BANNER = """
[bold cyan]
    _    ___ ____  ___    _      ____  _____ _   _ _____ ___ _   _ _____ _
   / \\  |_ _|  _ \\|_ _|  / \\    / ___|| ____| \\ | |_   _|_ _| \\ | | ____| |
  / _ \\  | || |_) || |  / _ \\   \\___ \\|  _| |  \\| | | |  | ||  \\| |  _| | |
 / ___ \\ | ||  _ < | | / ___ \\   ___) | |___| |\\  | | |  | || |\\  | |___| |___
/_/   \\_\\___|_| \\_\\___/_/   \\_\\ |____/|_____|_| \\_| |_| |___|_| \\_|_____|_____|
[/]
[dim]Multi-Agent Orchestration Platform — Treasury · Meta-Exchange · Organizer · JARVIS · Matrix[/]
[dim]Powered by Airia + LM Studio Cluster + Ollama + CCXT · Pipeline Engine (Domino/Vectorial/Matrix)[/]
"""


def main():
    console.print(BANNER)

    mode = sys.argv[1] if len(sys.argv) > 1 else "pipeline"

    if mode == "scan":
        asyncio.run(_scan())
    elif mode == "status":
        asyncio.run(_status())
    elif mode == "demo":
        asyncio.run(_demo())
    elif mode == "hitl":
        _hitl()
    elif mode == "demo-org":
        target = sys.argv[2] if len(sys.argv) > 2 else ""
        asyncio.run(_demo_org(target))
    elif mode == "demo-jarvis":
        asyncio.run(_demo_jarvis())
    elif mode == "demo-matrix":
        asyncio.run(_demo_matrix())
    elif mode == "demo-all":
        asyncio.run(_demo_all())
    elif mode == "meta":
        prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        asyncio.run(_meta(prompt))
    elif mode == "organize":
        target = sys.argv[2] if len(sys.argv) > 2 else ""
        asyncio.run(_organize(target))
    elif mode == "jarvis":
        user_input = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        asyncio.run(_jarvis(user_input))
    elif mode == "dashboard":
        _dashboard()
    elif mode == "pipeline-v1":
        asyncio.run(_pipeline_v1())
    elif mode in ("pipeline", "pipeline-v2", "full", "run"):
        asyncio.run(_pipeline())
    else:
        console.print(f"[red]Unknown mode:[/] {mode}")
        console.print("Available: pipeline, pipeline-v1, pipeline-v2, dashboard, scan, status, demo, demo-org, demo-jarvis, demo-matrix, demo-all, hitl, meta, organize, jarvis")
        sys.exit(1)


async def _pipeline():
    from src.orchestrator import run_full_pipeline_v2
    await run_full_pipeline_v2()


async def _pipeline_v1():
    from src.orchestrator import run_full_pipeline
    await run_full_pipeline()


def _dashboard():
    from src.services.dashboard_ws import start_dashboard_server
    start_dashboard_server()


async def _scan():
    from src.orchestrator import run_scan_only
    await run_scan_only()


async def _status():
    from src.orchestrator import run_status
    await run_status()


async def _demo():
    from demo.demo_scenario import run_demo
    await run_demo()


def _hitl():
    from src.services.hitl_webhook import start_server
    start_server()


async def _demo_org(target_dir: str = ""):
    from demo.demo_organizer import run_demo
    await run_demo(target_dir)


async def _demo_jarvis():
    from demo.demo_jarvis import run_demo
    await run_demo()


async def _demo_matrix():
    from demo.demo_matrix import run_demo
    await run_demo()


async def _demo_all():
    """Run 3 hackathon demos: Sentinel + Organizer + JARVIS."""
    import time
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.align import Align
    from rich.text import Text

    console.clear()

    # ── Presentation intro ────────────────────────────────────────────────
    console.print(Panel(
        Align.center(Text.from_markup(
            "[bold cyan]"
            "    _    ___ ____  ___    _      ____  _____ _   _ _____ ___ _   _ _____ _\n"
            "   / \\  |_ _|  _ \\|_ _|  / \\    / ___|| ____| \\ | |_   _|_ _| \\ | | ____| |\n"
            "  / _ \\  | || |_) || |  / _ \\   \\___ \\|  _| |  \\| | | |  | ||  \\| |  _| | |\n"
            " / ___ \\ | ||  _ < | | / ___ \\   ___) | |___| |\\  | | |  | || |\\  | |___| |___\n"
            "/_/   \\_\\___|_| \\_\\___/_/   \\_\\ |____/|_____|_| \\_| |_| |___|_| \\_|_____|_____|\n"
            "[/]\n\n"
            "[bold white]Multi-Agent Orchestration Platform[/]\n"
            "[dim]Airia AI Agents Challenge — Track 2: Active Agents[/]"
        )),
        border_style="bright_cyan",
        padding=(1, 2),
    ))

    time.sleep(1)

    console.print(Panel(
        "[bold white]3 USE CASES — 1 PLATFORM[/]\n\n"
        "  [yellow]1.[/] [bold cyan]SENTINEL[/]     Treasury Risk Pipeline\n"
        "                    Multi-agent market scan, consensus IA, rapport PDF, HITL\n\n"
        "  [yellow]2.[/] [bold magenta]ORGANIZER[/]    Rangement Intelligent de Fichiers\n"
        "                    Scan, tri par categorie, deduplication, backup, scoring\n\n"
        "  [yellow]3.[/] [bold green]JARVIS[/]       Assistant Personnel Multi-Agent\n"
        "                    Classification d'intent, routage IA, fallback chains\n\n"
        "[dim]Stack: Airia SDK + LM Studio (5 GPU, 43GB VRAM) + Ollama + CCXT + SQLite[/]",
        title="[bold cyan]Hackathon Presentation[/]",
        border_style="cyan",
        padding=(0, 2),
    ))

    time.sleep(1.5)

    # ── Run 3 demos ───────────────────────────────────────────────────────
    demos = [
        ("SENTINEL", "Treasury Risk Pipeline", "demo.demo_scenario"),
        ("ORGANIZER", "Rangement Intelligent de Fichiers", "demo.demo_organizer"),
        ("JARVIS", "Assistant Personnel Multi-Agent", "demo.demo_jarvis"),
    ]

    results = []
    t_total = time.monotonic()

    for i, (name, desc, module_path) in enumerate(demos, 1):
        console.print()
        console.print(Panel(
            f"[bold]Use Case {i}/{len(demos)}[/]\n\n"
            f"[bold white]{desc}[/]",
            title=f"[bold cyan]{name}[/]",
            border_style="cyan",
            padding=(0, 2),
        ))
        time.sleep(0.5)

        t0 = time.monotonic()
        try:
            mod = __import__(module_path, fromlist=["run_demo"])
            if name == "ORGANIZER":
                await mod.run_demo("", clear_screen=False)
            else:
                await mod.run_demo(clear_screen=False)
            elapsed = time.monotonic() - t0
            results.append((name, "OK", elapsed))
            console.print(f"\n[green]{name} completed[/] in {elapsed:.1f}s\n")
        except Exception as e:
            elapsed = time.monotonic() - t0
            results.append((name, f"FAIL: {e}", elapsed))
            console.print(f"\n[red]{name} failed:[/] {e}\n")

    total_time = time.monotonic() - t_total

    # ── Final summary ─────────────────────────────────────────────────────
    console.print(Rule("[bold cyan]PRESENTATION COMPLETE[/]", style="cyan"))

    from rich.table import Table
    table = Table(title="Hackathon Demo Results", border_style="cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Use Case", style="cyan")
    table.add_column("Description")
    table.add_column("Status", justify="center")
    table.add_column("Time", justify="right")

    descs = {
        "SENTINEL": "Market scan + Consensus IA + PDF + HITL",
        "ORGANIZER": "Scan + Tri + Dedup + Backup + Scoring",
        "JARVIS": "Intent classifier + Multi-agent routing",
    }

    ok_count = 0
    for i, (name, status, elapsed) in enumerate(results, 1):
        is_ok = status == "OK"
        if is_ok:
            ok_count += 1
        table.add_row(
            str(i), name,
            descs.get(name, ""),
            "[green]OK[/]" if is_ok else f"[red]{status}[/]",
            f"{elapsed:.1f}s",
        )

    console.print(table)

    console.print(Panel(
        f"[bold]{ok_count}/{len(demos)} use cases completed[/] in [bold cyan]{total_time:.1f}s[/]\n\n"
        "[bold white]Platform Highlights:[/]\n"
        "  [cyan]>[/] 16+ AI agents across 5 groups\n"
        "  [cyan]>[/] Multi-model orchestration (LM Studio + Ollama + Airia)\n"
        "  [cyan]>[/] Pipeline Engine (Domino / Vectorial / Matrix patterns)\n"
        "  [cyan]>[/] Human-in-the-Loop approval gateway\n"
        "  [cyan]>[/] Full SQLite audit trail\n\n"
        "[dim italic]Airia Sentinel — From market signals to executive decisions, automatically.[/]",
        title="[bold green]Summary[/]",
        border_style="green",
        padding=(0, 2),
    ))

    console.print()
    console.print(Align.center(Text.from_markup(
        "[bold bright_cyan]AIRIA SENTINEL[/]  [dim]|[/]  "
        "[bold white]Multi-Agent Orchestration Platform[/]  [dim]|[/]  "
        "[bold green]Ready for Production[/]"
    )))
    console.print()


async def _meta(prompt: str = ""):
    from src.orchestrator import run_meta_exchange
    kwargs = {"prompt": prompt} if prompt else {}
    await run_meta_exchange(**kwargs)


async def _organize(target_dir: str = ""):
    from src.orchestrator import run_organizer
    kwargs = {"target_dir": target_dir} if target_dir else {}
    await run_organizer(**kwargs)


async def _jarvis(user_input: str = ""):
    from src.orchestrator import run_jarvis
    kwargs = {"user_input": user_input} if user_input else {}
    await run_jarvis(**kwargs)


if __name__ == "__main__":
    main()

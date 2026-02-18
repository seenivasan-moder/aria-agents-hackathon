#!/usr/bin/env python3
"""Airia Sentinel — CLI entry point.

Usage:
    python main.py                  # Full pipeline (default)
    python main.py scan             # Market scan only
    python main.py status           # Cluster health check
    python main.py demo             # Demo scenario for video
    python main.py hitl             # Start HITL webhook server
"""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console

console = Console()

BANNER = """
[bold cyan]
    _    ___ ____  ___    _      ____  _____ _   _ _____ ___ _   _ _____ _
   / \\  |_ _|  _ \\|_ _|  / \\    / ___|| ____| \\ | |_   _|_ _| \\ | | ____| |
  / _ \\  | || |_) || |  / _ \\   \\___ \\|  _| |  \\| | | |  | ||  \\| |  _| | |
 / ___ \\ | ||  _ < | | / ___ \\   ___) | |___| |\\  | | |  | || |\\  | |___| |___
/_/   \\_\\___|_| \\_\\___/_/   \\_\\ |____/|_____|_| \\_| |_| |___|_| \\_|_____|_____|
[/]
[dim]Multi-Agent Treasury Orchestration & Risk Management[/]
[dim]Powered by Airia + LM Studio Cluster + CCXT[/]
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
    elif mode in ("pipeline", "full", "run"):
        asyncio.run(_pipeline())
    else:
        console.print(f"[red]Unknown mode:[/] {mode}")
        console.print("Available: pipeline, scan, status, demo, hitl")
        sys.exit(1)


async def _pipeline():
    from src.orchestrator import run_full_pipeline
    await run_full_pipeline()


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


if __name__ == "__main__":
    main()

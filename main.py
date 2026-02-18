#!/usr/bin/env python3
"""Airia Sentinel — CLI entry point.

Usage:
    python main.py                  # Full pipeline (default)
    python main.py scan             # Market scan only
    python main.py status           # Cluster health check
    python main.py demo             # Demo scenario for video
    python main.py hitl             # Start HITL webhook server
    python main.py meta [prompt]    # Meta-Exchange: multi-AI orchestration
    python main.py organize [dir]   # Organizer: file scan + dedup
    python main.py jarvis [cmd]     # JARVIS: personal AI assistant
    python main.py demo-org [dir]   # Demo: Organizer (real file scan)
    python main.py demo-jarvis      # Demo: JARVIS (multi-agent routing)
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
[dim]Multi-Agent Orchestration Platform — Treasury · Meta-Exchange · Organizer · JARVIS[/]
[dim]Powered by Airia + LM Studio Cluster + Ollama + CCXT[/]
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
    elif mode == "meta":
        prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        asyncio.run(_meta(prompt))
    elif mode == "organize":
        target = sys.argv[2] if len(sys.argv) > 2 else ""
        asyncio.run(_organize(target))
    elif mode == "jarvis":
        user_input = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        asyncio.run(_jarvis(user_input))
    elif mode in ("pipeline", "full", "run"):
        asyncio.run(_pipeline())
    else:
        console.print(f"[red]Unknown mode:[/] {mode}")
        console.print("Available: pipeline, scan, status, demo, demo-org, demo-jarvis, hitl, meta, organize, jarvis")
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


async def _demo_org(target_dir: str = ""):
    from demo.demo_organizer import run_demo
    await run_demo(target_dir)


async def _demo_jarvis():
    from demo.demo_jarvis import run_demo
    await run_demo()


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

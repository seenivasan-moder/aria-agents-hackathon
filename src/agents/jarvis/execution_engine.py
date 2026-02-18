"""Agent Execution-Engine — Routes parsed intents to the appropriate sub-agent.

Central dispatcher that receives a ParsedIntent and executes it through
the correct sub-agent (ia-deep, ia-fast, ia-system, ia-trading, voice).
Manages tool availability, timeout handling, and fallback chains.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from src.config import config
from src.agents.jarvis.intent_classifier import ParsedIntent, IntentDomain, IntentAction
from src.services.lm_cluster import query_lm, query_ollama
from src.airia_bridge import bridge
from src import database as db

console = Console()


@dataclass
class ExecutionResult:
    success: bool
    response: str
    agent_used: str
    model_used: str = ""
    latency_ms: float = 0
    tool_calls: list[str] = field(default_factory=list)
    fallback_used: bool = False


# ── Sub-Agent Routing Table ──────────────────────────────────────────────

AGENT_ROUTES = {
    "ia-deep": {
        "description": "Deep analysis with qwen3-30b (complex reasoning, code review)",
        "backend": "lm_studio",
        "node": "M1",
        "timeout": 90,
    },
    "ia-fast": {
        "description": "Quick answers with qwen3:1.7b (simple questions, chat)",
        "backend": "ollama",
        "node": "OL1",
        "timeout": 15,
    },
    "ia-system": {
        "description": "System operations (launch apps, manage files)",
        "backend": "local",
        "node": None,
        "timeout": 10,
    },
    "ia-trading": {
        "description": "Trading analysis with market data",
        "backend": "lm_studio",
        "node": "M1",
        "timeout": 60,
    },
    "orchestrator": {
        "description": "Run Sentinel pipeline",
        "backend": "pipeline",
        "node": None,
        "timeout": 120,
    },
    "voice": {
        "description": "Voice control commands",
        "backend": "local",
        "node": None,
        "timeout": 5,
    },
}

FALLBACK_CHAIN = {
    "ia-deep": ["ia-fast", "airia"],
    "ia-fast": ["ia-deep"],
    "ia-trading": ["ia-deep", "ia-fast"],
    "ia-system": [],  # no fallback for system ops
    "orchestrator": [],
    "voice": [],
}


async def _execute_lm(intent: ParsedIntent, agent_config: dict) -> ExecutionResult:
    """Execute through LM Studio."""
    system_prompts = {
        "ia-deep": "Tu es un assistant expert. Reponds de maniere detaillee et structuree en francais.",
        "ia-trading": "Tu es un analyste de marche expert. Analyse les donnees et donne des recommandations precises.",
    }
    system = system_prompts.get(intent.suggested_agent, "")
    result = await query_lm(
        intent.raw_input,
        node_name=agent_config["node"],
        system=system,
    )
    return ExecutionResult(
        success=result.get("ok", False),
        response=result.get("content", result.get("error", "No response")),
        agent_used=intent.suggested_agent,
        model_used=result.get("model", ""),
        latency_ms=result.get("latency_ms", 0),
    )


async def _execute_ollama(intent: ParsedIntent, agent_config: dict) -> ExecutionResult:
    """Execute through Ollama."""
    result = await query_ollama(
        intent.raw_input,
        node_name=agent_config["node"],
        system="Tu es un assistant rapide. Reponds de maniere concise en francais.",
    )
    return ExecutionResult(
        success=result.get("ok", False),
        response=result.get("content", result.get("error", "No response")),
        agent_used=intent.suggested_agent,
        model_used=result.get("model", ""),
        latency_ms=result.get("latency_ms", 0),
    )


async def _execute_local(intent: ParsedIntent) -> ExecutionResult:
    """Execute local system operations."""
    if intent.action == IntentAction.OPEN and intent.entities.get("app"):
        app = intent.entities["app"]
        return ExecutionResult(
            success=True,
            response=f"Commande: lancer {intent.entities.get('app_name', app)}",
            agent_used="ia-system",
            tool_calls=[f"start {app}"],
        )

    if intent.action == IntentAction.STATUS:
        return ExecutionResult(
            success=True,
            response="Systeme operationnel. Cluster: M1 + OL1. Pipeline: pret.",
            agent_used="ia-system",
        )

    if intent.action == IntentAction.STOP:
        return ExecutionResult(
            success=True,
            response="Arret demande.",
            agent_used="voice",
        )

    return ExecutionResult(
        success=True,
        response=f"Action {intent.action.value} sur {intent.domain.value}: en attente d'implementation",
        agent_used="ia-system",
    )


async def execute(intent: ParsedIntent) -> ExecutionResult:
    """Route a parsed intent to the appropriate sub-agent and execute.

    Handles timeout and fallback chains automatically.
    """
    agent_name = intent.suggested_agent
    agent_config = AGENT_ROUTES.get(agent_name, AGENT_ROUTES["ia-fast"])

    # Local execution (no AI needed)
    if agent_config["backend"] == "local":
        return await _execute_local(intent)

    # Pipeline execution
    if agent_config["backend"] == "pipeline":
        return ExecutionResult(
            success=True,
            response="Lancement du pipeline Sentinel...",
            agent_used="orchestrator",
            tool_calls=["run_full_pipeline()"],
        )

    # AI execution with timeout and fallback
    try:
        if agent_config["backend"] == "lm_studio":
            result = await asyncio.wait_for(
                _execute_lm(intent, agent_config),
                timeout=agent_config["timeout"],
            )
        elif agent_config["backend"] == "ollama":
            result = await asyncio.wait_for(
                _execute_ollama(intent, agent_config),
                timeout=agent_config["timeout"],
            )
        else:
            result = ExecutionResult(False, "Unknown backend", agent_name)

        if result.success:
            return result

    except asyncio.TimeoutError:
        console.print(f"  [yellow]Timeout on {agent_name} ({agent_config['timeout']}s)[/]")
    except Exception as e:
        console.print(f"  [red]Error on {agent_name}: {e}[/]")

    # Try fallback chain
    for fallback_agent in FALLBACK_CHAIN.get(agent_name, []):
        console.print(f"  [dim]Trying fallback: {fallback_agent}[/]")
        fallback_config = AGENT_ROUTES.get(fallback_agent)
        if not fallback_config:
            continue

        try:
            intent.suggested_agent = fallback_agent
            if fallback_config["backend"] == "lm_studio":
                result = await asyncio.wait_for(
                    _execute_lm(intent, fallback_config),
                    timeout=fallback_config["timeout"],
                )
            elif fallback_config["backend"] == "ollama":
                result = await asyncio.wait_for(
                    _execute_ollama(intent, fallback_config),
                    timeout=fallback_config["timeout"],
                )
            else:
                continue

            if result.success:
                result.fallback_used = True
                return result
        except (asyncio.TimeoutError, Exception):
            continue

    return ExecutionResult(
        success=False,
        response="Tous les agents sont indisponibles.",
        agent_used=agent_name,
    )


async def run(run_id: str, intent: ParsedIntent) -> ExecutionResult:
    """Execute Agent Execution-Engine: dispatch intent to sub-agent."""
    t0 = time.monotonic()
    console.print(f"[bold magenta]Agent Execution-Engine[/] dispatching to {intent.suggested_agent}...")

    result = await execute(intent)
    result.latency_ms = (time.monotonic() - t0) * 1000

    db.save_audit(
        run_id, "execution", "execution_engine",
        input_summary=f"{intent.domain.value}/{intent.action.value} -> {intent.suggested_agent}",
        output_summary=f"{'OK' if result.success else 'FAIL'}, agent={result.agent_used}, "
                       f"model={result.model_used}, fallback={result.fallback_used}",
        model_used=result.model_used,
        latency_ms=result.latency_ms,
    )

    if result.success:
        console.print(f"[green]Execution done[/] — {result.agent_used} ({result.model_used or 'local'}) in {int(result.latency_ms)}ms")
    else:
        console.print(f"[red]Execution failed[/] — {result.response}")

    return result

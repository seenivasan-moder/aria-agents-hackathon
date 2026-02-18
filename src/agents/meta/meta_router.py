"""Agent Meta-Router — Intelligent AI provider selection and query routing.

Analyzes incoming prompts and routes them to the optimal AI provider based on:
- Query complexity (simple/medium/complex)
- Required capabilities (code, reasoning, web search, vision)
- Cost constraints and budget tracking
- Provider availability and latency history
- Context window requirements

Supported providers:
- LM Studio M1 (qwen3-30b): Deep analysis, complex reasoning
- LM Studio M2 (deepseek-coder-v2-lite): Code generation, fast code review
- Ollama OL1 (qwen3:1.7b): Quick answers, corrections, classification
- Ollama Cloud (minimax, glm, kimi): Web search, sub-agents
- Airia Platform: Pipeline orchestration, formal analysis
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rich.console import Console
from rich.table import Table

from src.config import config
from src.airia_bridge import bridge
from src import database as db

console = Console()


class Complexity(str, Enum):
    SIMPLE = "simple"      # < 50 tokens, factual, classification
    MEDIUM = "medium"      # 50-200 tokens, analysis, summarization
    COMPLEX = "complex"    # > 200 tokens, reasoning, multi-step


class Capability(str, Enum):
    CODE = "code"
    REASONING = "reasoning"
    WEB_SEARCH = "web_search"
    TRANSLATION = "translation"
    CLASSIFICATION = "classification"
    CREATIVE = "creative"


@dataclass
class ProviderProfile:
    name: str
    endpoint: str
    model: str
    capabilities: list[Capability]
    cost_per_1k_tokens: float  # USD
    avg_latency_ms: float = 0
    max_context: int = 32768
    online: bool = False
    priority: int = 0  # lower = higher priority


@dataclass
class RoutingDecision:
    provider: str
    model: str
    reason: str
    estimated_cost: float
    estimated_latency_ms: float
    complexity: Complexity
    capabilities_matched: list[str]
    fallback_provider: str = ""


# ── Provider Registry ─────────────────────────────────────────────────────

PROVIDERS = [
    ProviderProfile(
        "M1", "http://127.0.0.1:1234", "qwen3-30b",
        [Capability.REASONING, Capability.CODE, Capability.CREATIVE, Capability.TRANSLATION],
        cost_per_1k_tokens=0.0, max_context=32768, priority=1,
    ),
    ProviderProfile(
        "M2", "http://192.168.1.26:1234", "deepseek-coder-v2-lite",
        [Capability.CODE, Capability.REASONING],
        cost_per_1k_tokens=0.0, max_context=16384, priority=2,
    ),
    ProviderProfile(
        "OL1", "http://127.0.0.1:11434", "qwen3:1.7b",
        [Capability.CLASSIFICATION, Capability.TRANSLATION],
        cost_per_1k_tokens=0.0, max_context=8192, priority=3,
    ),
    ProviderProfile(
        "Ollama-Cloud", "http://127.0.0.1:11434", "minimax-m2.5:cloud",
        [Capability.WEB_SEARCH, Capability.REASONING, Capability.CREATIVE],
        cost_per_1k_tokens=0.001, max_context=128000, priority=4,
    ),
    ProviderProfile(
        "Airia", "https://platform.airia.com", "gpt-5.1",
        [Capability.REASONING, Capability.CREATIVE, Capability.CODE],
        cost_per_1k_tokens=0.005, max_context=128000, priority=5,
    ),
]


def _estimate_complexity(prompt: str) -> Complexity:
    """Estimate query complexity from prompt characteristics."""
    word_count = len(prompt.split())
    has_code = any(kw in prompt.lower() for kw in ["```", "def ", "class ", "function", "import "])
    has_reasoning = any(kw in prompt.lower() for kw in ["pourquoi", "explique", "analyse", "compare", "why", "explain"])
    has_multi_step = any(kw in prompt.lower() for kw in ["etape", "step", "d'abord", "ensuite", "first", "then"])

    if word_count < 20 and not has_code and not has_reasoning:
        return Complexity.SIMPLE
    if has_multi_step or (has_code and has_reasoning) or word_count > 100:
        return Complexity.COMPLEX
    return Complexity.MEDIUM


def _detect_capabilities(prompt: str) -> list[Capability]:
    """Detect required capabilities from prompt content."""
    caps = []
    lower = prompt.lower()

    if any(kw in lower for kw in ["code", "fonction", "class", "bug", "script", "python", "javascript"]):
        caps.append(Capability.CODE)
    if any(kw in lower for kw in ["pourquoi", "analyse", "explique", "raisonne", "compare", "evaluate"]):
        caps.append(Capability.REASONING)
    if any(kw in lower for kw in ["cherche", "google", "web", "actualite", "news", "search"]):
        caps.append(Capability.WEB_SEARCH)
    if any(kw in lower for kw in ["tradui", "translate", "anglais", "francais", "english"]):
        caps.append(Capability.TRANSLATION)
    if any(kw in lower for kw in ["classe", "categori", "tri", "classif", "detect"]):
        caps.append(Capability.CLASSIFICATION)
    if any(kw in lower for kw in ["ecri", "genere", "creat", "invente", "imagine", "write"]):
        caps.append(Capability.CREATIVE)

    return caps or [Capability.REASONING]


def route(prompt: str, budget_limit: float = 1.0) -> RoutingDecision:
    """Route a prompt to the optimal AI provider.

    Args:
        prompt: The user's query
        budget_limit: Maximum cost in USD for this query

    Returns:
        RoutingDecision with selected provider and reasoning
    """
    complexity = _estimate_complexity(prompt)
    required_caps = _detect_capabilities(prompt)
    estimated_tokens = len(prompt.split()) * 2  # rough estimate

    # Score each provider
    best_provider = None
    best_score = -1
    fallback = None

    for provider in PROVIDERS:
        # Capability match score
        cap_matches = [c for c in required_caps if c in provider.capabilities]
        cap_score = len(cap_matches) / max(len(required_caps), 1)

        # Complexity fit
        if complexity == Complexity.SIMPLE:
            complexity_score = 1.0 if provider.priority >= 3 else 0.5  # prefer light models
        elif complexity == Complexity.COMPLEX:
            complexity_score = 1.0 if provider.priority <= 2 else 0.3  # prefer heavy models
        else:
            complexity_score = 0.7

        # Cost score (prefer free local models)
        estimated_cost = (estimated_tokens / 1000) * provider.cost_per_1k_tokens
        cost_score = 1.0 if estimated_cost == 0 else max(0, 1 - estimated_cost / budget_limit)

        # Context fit
        context_score = 1.0 if estimated_tokens < provider.max_context else 0.0

        total_score = cap_score * 0.4 + complexity_score * 0.3 + cost_score * 0.2 + context_score * 0.1

        if total_score > best_score:
            fallback = best_provider
            best_score = total_score
            best_provider = provider

    if not best_provider:
        best_provider = PROVIDERS[0]  # fallback to M1

    estimated_cost = (estimated_tokens / 1000) * best_provider.cost_per_1k_tokens

    return RoutingDecision(
        provider=best_provider.name,
        model=best_provider.model,
        reason=f"Best match for {complexity.value} query requiring {', '.join(c.value for c in required_caps)}",
        estimated_cost=estimated_cost,
        estimated_latency_ms=best_provider.avg_latency_ms,
        complexity=complexity,
        capabilities_matched=[c.value for c in required_caps if c in best_provider.capabilities],
        fallback_provider=fallback.name if fallback else "M1",
    )


async def run(run_id: str, prompt: str) -> RoutingDecision:
    """Execute Agent Meta-Router: analyze prompt and select optimal provider."""
    t0 = time.monotonic()
    console.print("[bold magenta]Agent Meta-Router[/] analyzing prompt...")

    decision = route(prompt)
    latency = (time.monotonic() - t0) * 1000

    db.save_audit(
        run_id, "meta_routing", "meta_router",
        input_summary=f"{len(prompt)} chars, complexity={decision.complexity.value}",
        output_summary=f"Routed to {decision.provider} ({decision.model}), cost=${decision.estimated_cost:.4f}",
        model_used="rule-based",
        latency_ms=latency,
    )

    _print_decision(decision)
    console.print(f"[green]Meta-Router done[/] — {decision.provider} selected in {int(latency)}ms")
    return decision


def _print_decision(d: RoutingDecision) -> None:
    table = Table(title="Routing Decision", show_lines=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Provider", f"[bold]{d.provider}[/]")
    table.add_row("Model", d.model)
    table.add_row("Complexity", d.complexity.value)
    table.add_row("Capabilities", ", ".join(d.capabilities_matched))
    table.add_row("Est. Cost", f"${d.estimated_cost:.4f}")
    table.add_row("Reason", d.reason)
    table.add_row("Fallback", d.fallback_provider)
    console.print(table)

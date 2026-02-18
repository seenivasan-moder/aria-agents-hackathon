"""Agent Intent-Classifier — Parse voice/text input and detect user intent.

Classifies user commands into domains (system, web, trading, analysis, voice)
and extracts entities (app names, URLs, file paths, trading pairs).
Uses fast local IA (Ollama qwen3:1.7b) for ambiguous cases.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rich.console import Console
from rich.table import Table

from src.config import config
from src import database as db

console = Console()


class IntentDomain(str, Enum):
    SYSTEM = "system"        # Launch apps, manage files, run scripts
    WEB = "web"              # Browser, search, URLs
    TRADING = "trading"      # Market scan, positions, signals
    ANALYSIS = "analysis"    # Deep reasoning, code review, explain
    VOICE = "voice"          # Voice control (stop, repeat, louder)
    CONVERSATION = "conversation"  # Casual chat, questions
    PIPELINE = "pipeline"    # Run agent pipelines


class IntentAction(str, Enum):
    OPEN = "open"
    CLOSE = "close"
    SEARCH = "search"
    ANALYZE = "analyze"
    SCAN = "scan"
    EXECUTE = "execute"
    STATUS = "status"
    NAVIGATE = "navigate"
    CREATE = "create"
    DELETE = "delete"
    EXPLAIN = "explain"
    STOP = "stop"
    CHAT = "chat"


@dataclass
class ParsedIntent:
    domain: IntentDomain
    action: IntentAction
    confidence: float  # 0-100
    entities: dict[str, str] = field(default_factory=dict)
    raw_input: str = ""
    requires_ai: bool = False  # needs AI for execution
    suggested_agent: str = ""  # which sub-agent should handle this


# ── Intent Rules ─────────────────────────────────────────────────────

SYSTEM_KEYWORDS = {
    "ouvre": IntentAction.OPEN, "lance": IntentAction.OPEN, "demarre": IntentAction.OPEN,
    "open": IntentAction.OPEN, "launch": IntentAction.OPEN, "start": IntentAction.OPEN,
    "ferme": IntentAction.CLOSE, "close": IntentAction.CLOSE, "kill": IntentAction.CLOSE,
    "arrete": IntentAction.STOP, "stop": IntentAction.STOP,
    "cree": IntentAction.CREATE, "create": IntentAction.CREATE,
    "supprime": IntentAction.DELETE, "delete": IntentAction.DELETE,
    "status": IntentAction.STATUS, "etat": IntentAction.STATUS,
}

WEB_KEYWORDS = ["cherche", "google", "search", "web", "youtube", "github", "http", "www", "site"]
TRADING_KEYWORDS = ["trade", "trading", "marche", "market", "bitcoin", "btc", "eth", "crypto", "forex", "scan"]
ANALYSIS_KEYWORDS = ["analyse", "analyze", "explique", "explain", "review", "code", "debug", "pourquoi", "why"]
VOICE_KEYWORDS = ["repete", "repeat", "plus fort", "louder", "silence", "mute", "jarvis"]
PIPELINE_KEYWORDS = ["pipeline", "sentinel", "agents", "full scan", "rapport", "report"]

APP_ENTITIES = {
    "chrome": "chrome.exe", "firefox": "firefox.exe", "code": "code.exe",
    "vscode": "code.exe", "vs code": "code.exe", "terminal": "cmd.exe",
    "explorateur": "explorer.exe", "explorer": "explorer.exe",
    "notepad": "notepad.exe", "spotify": "spotify.exe",
    "discord": "discord.exe", "teams": "teams.exe", "slack": "slack.exe",
}


def _extract_entities(text: str) -> dict[str, str]:
    """Extract named entities from text."""
    entities = {}
    lower = text.lower()

    # App names
    for app_name, exe in APP_ENTITIES.items():
        if app_name in lower:
            entities["app"] = exe
            entities["app_name"] = app_name
            break

    # URLs
    url_match = re.search(r'https?://[^\s]+', text)
    if url_match:
        entities["url"] = url_match.group()

    # File paths
    path_match = re.search(r'[A-Z]:\\[^\s]+|/[^\s]+\.\w+', text)
    if path_match:
        entities["path"] = path_match.group()

    # Trading pairs
    pair_match = re.search(r'\b(BTC|ETH|SOL|EUR|GBP|JPY|XAU|XAG)/?(USDT|USD|EUR)?\b', text.upper())
    if pair_match:
        entities["pair"] = pair_match.group()

    return entities


def classify(text: str) -> ParsedIntent:
    """Classify user input into domain, action, and entities.

    Uses rule-based classification for speed (< 1ms).
    Falls back to AI for ambiguous cases.
    """
    lower = text.lower().strip()
    words = lower.split()
    entities = _extract_entities(text)

    # Check each domain by keyword matching
    # Priority: system > trading > web > pipeline > analysis > voice > conversation

    # System commands
    for keyword, action in SYSTEM_KEYWORDS.items():
        if keyword in words:
            return ParsedIntent(
                domain=IntentDomain.SYSTEM, action=action, confidence=90,
                entities=entities, raw_input=text, suggested_agent="ia-system",
            )

    # Trading
    if any(kw in lower for kw in TRADING_KEYWORDS):
        action = IntentAction.SCAN if "scan" in lower else IntentAction.ANALYZE
        return ParsedIntent(
            domain=IntentDomain.TRADING, action=action, confidence=85,
            entities=entities, raw_input=text, requires_ai=True,
            suggested_agent="ia-trading",
        )

    # Web
    if any(kw in lower for kw in WEB_KEYWORDS) or entities.get("url"):
        action = IntentAction.SEARCH if "cherche" in lower or "search" in lower else IntentAction.NAVIGATE
        return ParsedIntent(
            domain=IntentDomain.WEB, action=action, confidence=85,
            entities=entities, raw_input=text, suggested_agent="ia-system",
        )

    # Pipeline
    if any(kw in lower for kw in PIPELINE_KEYWORDS):
        return ParsedIntent(
            domain=IntentDomain.PIPELINE, action=IntentAction.EXECUTE, confidence=90,
            entities=entities, raw_input=text, suggested_agent="orchestrator",
        )

    # Analysis (needs AI)
    if any(kw in lower for kw in ANALYSIS_KEYWORDS):
        action = IntentAction.EXPLAIN if "explique" in lower or "explain" in lower else IntentAction.ANALYZE
        return ParsedIntent(
            domain=IntentDomain.ANALYSIS, action=action, confidence=80,
            entities=entities, raw_input=text, requires_ai=True,
            suggested_agent="ia-deep",
        )

    # Voice control
    if any(kw in lower for kw in VOICE_KEYWORDS):
        return ParsedIntent(
            domain=IntentDomain.VOICE, action=IntentAction.EXECUTE, confidence=95,
            entities=entities, raw_input=text, suggested_agent="voice",
        )

    # Default: conversation (needs AI)
    return ParsedIntent(
        domain=IntentDomain.CONVERSATION, action=IntentAction.CHAT, confidence=60,
        entities=entities, raw_input=text, requires_ai=True,
        suggested_agent="ia-fast",
    )


async def run(run_id: str, user_input: str) -> ParsedIntent:
    """Execute Agent Intent-Classifier: parse input and detect intent."""
    t0 = time.monotonic()
    console.print("[bold magenta]Agent Intent-Classifier[/] parsing input...")

    intent = classify(user_input)
    latency = (time.monotonic() - t0) * 1000

    db.save_audit(
        run_id, "intent_classification", "intent_classifier",
        input_summary=f"'{user_input[:80]}...' ({len(user_input)} chars)",
        output_summary=f"domain={intent.domain.value}, action={intent.action.value}, "
                       f"confidence={intent.confidence}, agent={intent.suggested_agent}",
        model_used="rule-based",
        latency_ms=latency,
    )

    _print_intent(intent)
    console.print(f"[green]Intent-Classifier done[/] — {intent.domain.value}/{intent.action.value} in {int(latency)}ms")
    return intent


def _print_intent(intent: ParsedIntent) -> None:
    table = Table(title="Parsed Intent", show_lines=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Domain", f"[bold]{intent.domain.value}[/]")
    table.add_row("Action", intent.action.value)
    table.add_row("Confidence", f"{intent.confidence:.0f}%")
    table.add_row("Suggested Agent", intent.suggested_agent)
    table.add_row("Requires AI", "[yellow]Yes[/]" if intent.requires_ai else "[green]No[/]")
    if intent.entities:
        table.add_row("Entities", ", ".join(f"{k}={v}" for k, v in intent.entities.items()))
    console.print(table)

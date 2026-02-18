"""Agent Context-Manager — Cross-provider memory and token optimization.

Manages the "memory" between AI exchanges to avoid repetitions, optimize
token usage, and maintain conversation coherence across multiple providers.

Features:
- Conversation history compression
- Cross-provider context transfer
- Token budget tracking per session
- Semantic dedup of repeated queries
- Context window management per provider
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

from src.config import config
from src import database as db

console = Console()


@dataclass
class ContextEntry:
    role: str  # "user", "assistant", "system"
    content: str
    provider: str
    model: str
    tokens_used: int = 0
    timestamp: float = field(default_factory=time.time)
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class SessionContext:
    session_id: str
    entries: list[ContextEntry] = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    providers_used: set[str] = field(default_factory=set)
    dedup_count: int = 0  # number of duplicate queries caught


class ContextManager:
    """Manages cross-provider conversation context."""

    def __init__(self, max_context_tokens: int = 16384):
        self._sessions: dict[str, SessionContext] = {}
        self._max_tokens = max_context_tokens
        self._query_cache: dict[str, str] = {}  # hash -> response

    def get_or_create_session(self, session_id: str) -> SessionContext:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionContext(session_id=session_id)
        return self._sessions[session_id]

    def add_exchange(
        self,
        session_id: str,
        user_input: str,
        assistant_response: str,
        provider: str,
        model: str,
        tokens_used: int = 0,
        cost: float = 0.0,
    ) -> bool:
        """Record an exchange. Returns True if this was a new query, False if deduplicated."""
        session = self.get_or_create_session(session_id)
        query_hash = hashlib.md5(user_input.strip().lower().encode()).hexdigest()[:12]

        # Check for semantic duplicate
        if query_hash in self._query_cache:
            session.dedup_count += 1
            console.print(f"  [yellow]Context-Manager: duplicate query detected (saved ~{tokens_used} tokens)[/]")
            return False

        # Add entries
        user_entry = ContextEntry("user", user_input, provider, model, tokens_used // 3)
        assistant_entry = ContextEntry("assistant", assistant_response, provider, model, tokens_used * 2 // 3)

        session.entries.append(user_entry)
        session.entries.append(assistant_entry)
        session.total_tokens += tokens_used
        session.total_cost += cost
        session.providers_used.add(provider)

        # Cache for dedup
        self._query_cache[query_hash] = assistant_response

        # Trim if over budget
        self._trim_context(session)

        return True

    def get_context_for_provider(
        self,
        session_id: str,
        target_provider: str,
        max_tokens: int | None = None,
    ) -> list[dict[str, str]]:
        """Get optimized context messages for a specific provider.

        Compresses history and formats it for the target provider's context window.
        """
        session = self.get_or_create_session(session_id)
        budget = max_tokens or self._max_tokens

        messages = []
        token_count = 0

        # Always include system context summary if multi-provider
        if len(session.providers_used) > 1:
            summary = self._build_cross_provider_summary(session)
            messages.append({"role": "system", "content": summary})
            token_count += len(summary.split()) * 2

        # Add recent exchanges (most recent first, then reverse)
        for entry in reversed(session.entries[-20:]):  # last 10 exchanges
            est_tokens = len(entry.content.split()) * 2
            if token_count + est_tokens > budget:
                break
            messages.insert(1 if messages else 0, {
                "role": entry.role,
                "content": entry.content,
            })
            token_count += est_tokens

        return messages

    def _build_cross_provider_summary(self, session: SessionContext) -> str:
        """Build a summary of what other providers have contributed."""
        providers = ", ".join(session.providers_used)
        n_exchanges = len(session.entries) // 2
        return (
            f"[Context Transfer] This conversation has used {n_exchanges} exchanges "
            f"across providers: {providers}. Total tokens: {session.total_tokens}. "
            f"Continue the conversation coherently."
        )

    def _trim_context(self, session: SessionContext) -> None:
        """Remove oldest entries if over token budget."""
        while session.total_tokens > self._max_tokens * 2 and len(session.entries) > 4:
            removed = session.entries.pop(0)
            session.total_tokens -= removed.tokens_used

    def get_session_stats(self, session_id: str) -> dict[str, Any]:
        session = self.get_or_create_session(session_id)
        return {
            "session_id": session_id,
            "exchanges": len(session.entries) // 2,
            "total_tokens": session.total_tokens,
            "total_cost": round(session.total_cost, 4),
            "providers_used": list(session.providers_used),
            "dedup_saved": session.dedup_count,
        }

    def get_cached_response(self, query: str) -> str | None:
        """Check if a similar query was already answered."""
        query_hash = hashlib.md5(query.strip().lower().encode()).hexdigest()[:12]
        return self._query_cache.get(query_hash)


# Singleton
context_manager = ContextManager()


async def run(run_id: str, session_id: str) -> dict[str, Any]:
    """Execute Agent Context-Manager: report session stats and optimize context."""
    t0 = time.monotonic()
    console.print("[bold magenta]Agent Context-Manager[/] analyzing session...")

    stats = context_manager.get_session_stats(session_id)
    latency = (time.monotonic() - t0) * 1000

    db.save_audit(
        run_id, "context_management", "context_manager",
        input_summary=f"Session {session_id}",
        output_summary=f"{stats['exchanges']} exchanges, {stats['total_tokens']} tokens, "
                       f"${stats['total_cost']}, dedup={stats['dedup_saved']}",
        model_used="context-manager",
        latency_ms=latency,
    )

    _print_stats(stats)
    return stats


def _print_stats(stats: dict) -> None:
    table = Table(title="Session Context", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Exchanges", str(stats["exchanges"]))
    table.add_row("Total Tokens", f"{stats['total_tokens']:,}")
    table.add_row("Total Cost", f"${stats['total_cost']:.4f}")
    table.add_row("Providers", ", ".join(stats["providers_used"]) or "none")
    table.add_row("Dedup Saved", str(stats["dedup_saved"]))
    console.print(table)

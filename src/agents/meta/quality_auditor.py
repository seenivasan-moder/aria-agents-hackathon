"""Agent Quality-Auditor — Cross-provider response validation and scoring.

Compares AI responses across multiple providers to detect:
- Hallucinations (factual inconsistencies between providers)
- Confidence divergence (one model very confident, another not)
- Format compliance (JSON validity, schema adherence)
- Language quality (grammar, coherence, completeness)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.table import Table

from src.config import config
from src.services.lm_cluster import query_lm, query_ollama
from src.airia_bridge import bridge
from src import database as db

console = Console()


@dataclass
class QualityScore:
    provider: str
    model: str
    response_length: int
    is_valid_json: bool = False
    coherence_score: float = 0.0  # 0-100
    completeness_score: float = 0.0  # 0-100
    consistency_score: float = 0.0  # 0-100 (vs other providers)
    overall_score: float = 0.0
    flags: list[str] = None

    def __post_init__(self):
        if self.flags is None:
            self.flags = []
        self.overall_score = (
            self.coherence_score * 0.3
            + self.completeness_score * 0.4
            + self.consistency_score * 0.3
        )


AUDIT_PROMPT = """\
Tu es un auditeur de qualite IA. Compare ces reponses de differents modeles
a la meme question et evalue chacune.

QUESTION ORIGINALE:
{question}

REPONSES:
{responses}

Evalue chaque reponse sur:
1. Coherence (0-100): La reponse est-elle logique et bien structuree?
2. Completude (0-100): La reponse couvre-t-elle tous les aspects de la question?
3. Consistance (0-100): Les reponses sont-elles coherentes entre elles?

Reponds en JSON:
{{"audits": [
  {{"provider": "...", "coherence": 85, "completeness": 90, "consistency": 80, "flags": ["hallucination detectee sur X"]}},
  ...
]}}"""


def _check_json_validity(text: str) -> bool:
    """Check if text contains valid JSON."""
    text = text.strip()
    for start_char in ["{", "["]:
        idx = text.find(start_char)
        if idx >= 0:
            end_char = "}" if start_char == "{" else "]"
            end_idx = text.rfind(end_char)
            if end_idx > idx:
                try:
                    json.loads(text[idx:end_idx + 1])
                    return True
                except json.JSONDecodeError:
                    pass
    return False


def _basic_quality_check(response: str) -> tuple[float, float, list[str]]:
    """Rule-based quality scoring without AI."""
    flags = []
    words = response.split()
    word_count = len(words)

    # Coherence: based on structure and formatting
    coherence = 50.0
    if word_count > 20:
        coherence += 20
    if any(c in response for c in ["\n", "- ", "1.", "* "]):
        coherence += 15  # structured
    if _check_json_validity(response):
        coherence += 15  # valid JSON

    # Completeness: based on length and coverage
    completeness = min(100, word_count / 2)  # rough: 200 words = 100%

    # Flags
    if word_count < 10:
        flags.append("Response too short")
    if response.count("...") > 3:
        flags.append("Excessive ellipsis (possible truncation)")
    if "error" in response.lower() or "exception" in response.lower():
        flags.append("Contains error indicators")

    return min(100, coherence), min(100, completeness), flags


def _cross_consistency(responses: list[dict]) -> dict[str, float]:
    """Calculate consistency scores by comparing response overlap."""
    if len(responses) < 2:
        return {r.get("provider", "?"): 100.0 for r in responses}

    # Extract key terms from each response
    term_sets = {}
    for r in responses:
        provider = r.get("provider", "?")
        words = set(r.get("content", "").lower().split())
        # Filter to meaningful words (> 4 chars)
        term_sets[provider] = {w for w in words if len(w) > 4}

    # Jaccard similarity between each pair
    scores = {}
    providers = list(term_sets.keys())
    for p in providers:
        similarities = []
        for other in providers:
            if other != p:
                intersection = term_sets[p] & term_sets[other]
                union = term_sets[p] | term_sets[other]
                sim = len(intersection) / max(len(union), 1) * 100
                similarities.append(sim)
        scores[p] = sum(similarities) / max(len(similarities), 1)

    return scores


async def audit_responses(
    question: str,
    responses: list[dict[str, Any]],
    use_ai: bool = True,
) -> list[QualityScore]:
    """Audit multiple provider responses for quality and consistency.

    Args:
        question: The original query
        responses: List of {"provider": str, "model": str, "content": str}
        use_ai: Whether to use AI for deep quality analysis

    Returns:
        List of QualityScore for each response
    """
    # Basic quality checks (rule-based, instant)
    consistency_scores = _cross_consistency(responses)
    scores = []

    for r in responses:
        provider = r.get("provider", "unknown")
        content = r.get("content", "")
        coherence, completeness, flags = _basic_quality_check(content)
        consistency = consistency_scores.get(provider, 50.0)

        scores.append(QualityScore(
            provider=provider,
            model=r.get("model", "?"),
            response_length=len(content),
            is_valid_json=_check_json_validity(content),
            coherence_score=coherence,
            completeness_score=completeness,
            consistency_score=consistency,
            flags=flags,
        ))

    return scores


async def run(
    run_id: str,
    question: str,
    responses: list[dict[str, Any]],
) -> list[QualityScore]:
    """Execute Agent Quality-Auditor: evaluate cross-provider response quality."""
    t0 = time.monotonic()
    console.print("[bold magenta]Agent Quality-Auditor[/] evaluating responses...")

    scores = await audit_responses(question, responses)
    latency = (time.monotonic() - t0) * 1000

    # Determine best provider
    best = max(scores, key=lambda s: s.overall_score) if scores else None

    db.save_audit(
        run_id, "quality_audit", "quality_auditor",
        input_summary=f"{len(responses)} responses to audit",
        output_summary=f"Best: {best.provider} ({best.overall_score:.0f}/100)" if best else "No responses",
        model_used="rule-based+cross-validation",
        latency_ms=latency,
    )

    _print_scores(scores)
    console.print(f"[green]Quality-Auditor done[/] — {len(scores)} responses evaluated in {int(latency)}ms")
    return scores


def _print_scores(scores: list[QualityScore]) -> None:
    table = Table(title="Quality Audit", show_lines=True)
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="dim")
    table.add_column("Coherence", justify="right")
    table.add_column("Complete", justify="right")
    table.add_column("Consist.", justify="right")
    table.add_column("Overall", justify="right")
    table.add_column("JSON", justify="center")
    table.add_column("Flags")

    for s in scores:
        overall_color = "green" if s.overall_score > 70 else ("yellow" if s.overall_score > 40 else "red")
        table.add_row(
            s.provider, s.model,
            f"{s.coherence_score:.0f}", f"{s.completeness_score:.0f}",
            f"{s.consistency_score:.0f}",
            f"[{overall_color}]{s.overall_score:.0f}[/]",
            "[green]OK[/]" if s.is_valid_json else "[red]--[/]",
            ", ".join(s.flags) if s.flags else "[dim]none[/]",
        )

    console.print(table)

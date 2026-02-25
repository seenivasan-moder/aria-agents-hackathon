"""Agent Sentinel — News Scanner: financial news scanning and market-moving event detection.

Simulates a real-time news feed for each watched pair, classifying headlines by:
- Category: earnings, macro, geopolitical, regulatory, technical, sentiment
- Impact level: high, medium, low (keyword + category based)
- Sentiment: positive/negative/neutral with score (-100 to +100)

Features:
- Aggregate news sentiment per asset class
- Breaking news detection (high-impact + recent timestamp)
- Event clustering: group related news items by topic
- News-driven risk adjustment suggestions
- Airia enrichment: send top headlines to bridge for AI-powered sentiment analysis

Uses numpy for statistical aggregation.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.pipeline_engine import StepResult, StepStatus
from src.config import config
from src.airia_bridge import bridge
from src import database as db

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MAX_HEADLINES_PER_ASSET = 5         # Simulated headlines per watched pair
BREAKING_NEWS_WINDOW_MIN = 30      # Minutes — news within this window = "breaking"
MAX_NEWS_DISPLAY = 20              # Max headlines to show in Rich table
CLUSTER_SIMILARITY_THRESHOLD = 2   # Min shared keywords for clustering

# Keyword dictionaries for impact and sentiment classification
HIGH_IMPACT_KEYWORDS = [
    "crash", "surge", "halt", "ban", "default", "bankruptcy", "war",
    "rate hike", "rate cut", "emergency", "sanctions", "hack", "exploit",
    "record high", "record low", "collapse", "bailout", "liquidity crisis",
    "flash crash", "contagion", "systemic", "inflation spike", "recession",
]

MEDIUM_IMPACT_KEYWORDS = [
    "earnings beat", "earnings miss", "downgrade", "upgrade", "merger",
    "acquisition", "regulation", "investigation", "lawsuit", "partnership",
    "adoption", "whale", "outflow", "inflow", "etf approval", "etf rejection",
    "fed minutes", "ecb meeting", "boj decision", "gdp", "unemployment",
]

POSITIVE_KEYWORDS = [
    "surge", "rally", "bullish", "growth", "recovery", "upgrade", "adoption",
    "partnership", "approval", "inflow", "beat", "record high", "breakout",
    "accumulation", "milestone", "innovation", "expansion", "profit",
    "optimism", "confidence", "strong", "outperform", "boost",
]

NEGATIVE_KEYWORDS = [
    "crash", "plunge", "bearish", "decline", "recession", "downgrade", "ban",
    "hack", "exploit", "outflow", "miss", "record low", "breakdown",
    "liquidation", "warning", "risk", "default", "sanctions", "collapse",
    "pessimism", "fear", "weak", "underperform", "loss", "crisis",
]

# Category definitions
NEWS_CATEGORIES = ["earnings", "macro", "geopolitical", "regulatory", "technical", "sentiment"]

# Simulated news templates per asset class
_CRYPTO_HEADLINES = [
    ("BTC mining difficulty reaches new all-time high", "technical", "neutral"),
    ("Major exchange reports significant whale accumulation", "sentiment", "positive"),
    ("Regulatory framework proposal gains bipartisan support", "regulatory", "positive"),
    ("DeFi protocol exploit drains $50M in liquidity", "technical", "negative"),
    ("Central bank digital currency pilot expands to 10 countries", "macro", "neutral"),
    ("Institutional inflow hits quarterly record", "sentiment", "positive"),
    ("Flash crash triggered by cascading liquidations", "technical", "negative"),
    ("ETF approval sparks renewed optimism", "regulatory", "positive"),
    ("Mining ban extended to three additional provinces", "regulatory", "negative"),
    ("On-chain metrics signal strong accumulation phase", "technical", "positive"),
    ("Cross-chain bridge exploit raises security concerns", "technical", "negative"),
    ("Stablecoin depegging event triggers market fear", "sentiment", "negative"),
]

_FOREX_HEADLINES = [
    ("Fed signals potential rate cut at next meeting", "macro", "positive"),
    ("ECB holds rates steady amid inflation concerns", "macro", "neutral"),
    ("BOJ surprises markets with yield curve adjustment", "macro", "negative"),
    ("Trade deficit widens beyond analyst expectations", "macro", "negative"),
    ("Employment data beats consensus forecast", "earnings", "positive"),
    ("Currency intervention by central bank stabilizes pair", "geopolitical", "neutral"),
    ("Geopolitical tensions drive safe-haven flows", "geopolitical", "negative"),
    ("GDP growth surpasses expectations at 3.2%", "macro", "positive"),
    ("Inflation spike forces emergency policy review", "macro", "negative"),
    ("Cross-border payment partnership boosts volume", "technical", "positive"),
]

_COMMODITY_HEADLINES = [
    ("OPEC announces production cut extension", "geopolitical", "positive"),
    ("Gold surges on recession fears and risk-off sentiment", "macro", "positive"),
    ("Silver demand from solar industry hits record", "earnings", "positive"),
    ("Oil inventory build exceeds forecast by 5M barrels", "macro", "negative"),
    ("Natural gas supply disruption from pipeline incident", "geopolitical", "positive"),
    ("Commodity index rebalancing triggers large flows", "technical", "neutral"),
    ("Sanctions on major producer tighten supply outlook", "geopolitical", "positive"),
    ("Global demand weakness weighs on commodity complex", "macro", "negative"),
    ("Weather disruption threatens agricultural output", "geopolitical", "negative"),
    ("Strategic reserve release announced to cap prices", "regulatory", "negative"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NewsItem:
    """A single news headline with classification."""
    headline: str
    symbol: str
    category: str                   # earnings, macro, geopolitical, regulatory, technical, sentiment
    impact: str = "low"             # high, medium, low
    sentiment: str = "neutral"      # positive, negative, neutral
    sentiment_score: float = 0.0    # -100 to +100
    timestamp: str = ""
    is_breaking: bool = False
    cluster_id: int = -1            # Cluster assignment (-1 = unclustered)
    keywords: list[str] = field(default_factory=list)


@dataclass
class NewsScanReport:
    """Complete news scanning report."""
    items: list[NewsItem] = field(default_factory=list)
    total_headlines: int = 0
    breaking_count: int = 0
    high_impact_count: int = 0
    aggregate_sentiment: dict[str, float] = field(default_factory=dict)  # asset_class -> avg sentiment
    global_sentiment: float = 0.0
    clusters: list[dict] = field(default_factory=list)       # [{topic, headlines, avg_sentiment}]
    risk_adjustments: list[dict] = field(default_factory=list)
    airia_enriched: bool = False
    models_used: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# NEWS GENERATION — Simulated feed
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_asset_class(symbol: str) -> str:
    """Determine asset class from symbol."""
    sym = symbol.upper()
    if any(c in sym for c in ["BTC", "ETH", "SOL", "DOGE", "XRP", "ADA", "AVAX", "LINK", "SUI", "PEPE"]):
        return "crypto"
    if any(c in sym for c in ["EUR", "GBP", "JPY", "USD", "CHF", "AUD", "CAD", "NZD"]):
        return "forex"
    if any(c in sym for c in ["XAU", "XAG", "CL", "NG", "OIL", "GOLD", "SILVER"]):
        return "commodity"
    return "other"


def _get_headlines_pool(asset_class: str) -> list[tuple[str, str, str]]:
    """Get the headline template pool for an asset class."""
    pools = {
        "crypto": _CRYPTO_HEADLINES,
        "forex": _FOREX_HEADLINES,
        "commodity": _COMMODITY_HEADLINES,
    }
    return pools.get(asset_class, _FOREX_HEADLINES)


def _classify_impact(headline: str, category: str) -> str:
    """Classify headline impact level based on keywords and category."""
    hl_lower = headline.lower()

    # Check high-impact keywords
    for kw in HIGH_IMPACT_KEYWORDS:
        if kw in hl_lower:
            return "high"

    # Check medium-impact keywords
    for kw in MEDIUM_IMPACT_KEYWORDS:
        if kw in hl_lower:
            return "medium"

    # Category-based defaults
    if category in ("geopolitical", "regulatory"):
        return "medium"

    return "low"


def _compute_sentiment_score(headline: str, base_sentiment: str) -> tuple[str, float]:
    """Compute a sentiment score from headline text and base sentiment.

    Returns:
        (sentiment_label, score) where score is in [-100, +100]
    """
    hl_lower = headline.lower()

    pos_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw in hl_lower)
    neg_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw in hl_lower)

    # Base score from template sentiment
    base_scores = {"positive": 35.0, "negative": -35.0, "neutral": 0.0}
    score = base_scores.get(base_sentiment, 0.0)

    # Adjust by keyword hits
    score += pos_hits * 15.0
    score -= neg_hits * 15.0

    # Clamp to [-100, +100]
    score = float(np.clip(score, -100, 100))

    # Determine label
    if score > 15:
        label = "positive"
    elif score < -15:
        label = "negative"
    else:
        label = "neutral"

    return label, round(score, 1)


def _extract_keywords(headline: str) -> list[str]:
    """Extract relevant keywords from a headline for clustering."""
    hl_lower = headline.lower()
    all_keywords = POSITIVE_KEYWORDS + NEGATIVE_KEYWORDS + HIGH_IMPACT_KEYWORDS + MEDIUM_IMPACT_KEYWORDS
    found = [kw for kw in all_keywords if kw in hl_lower]
    return found


def _generate_news_feed(
    signals: list[dict],
    seed: int = 42,
) -> list[NewsItem]:
    """Generate a simulated news feed for all watched pairs.

    Uses signal data to seed randomness for reproducible but varied headlines.
    """
    rng = np.random.default_rng(seed)
    now = datetime.now(timezone.utc)
    items: list[NewsItem] = []

    for sig in signals:
        symbol = sig.get("symbol", "UNKNOWN")
        asset_class = _classify_asset_class(symbol)
        pool = _get_headlines_pool(asset_class)

        if not pool:
            continue

        # Select a subset of headlines for this symbol
        n_headlines = min(MAX_HEADLINES_PER_ASSET, len(pool))
        indices = rng.choice(len(pool), size=n_headlines, replace=False)

        for idx in indices:
            template_headline, category, base_sentiment = pool[idx]

            # Personalize headline with symbol
            headline = f"[{symbol}] {template_headline}"

            # Classify impact
            impact = _classify_impact(headline, category)

            # Compute sentiment score
            sentiment_label, sentiment_score = _compute_sentiment_score(headline, base_sentiment)

            # Adjust sentiment based on actual signal direction
            direction = sig.get("direction", "neutral")
            change_24h = sig.get("change_24h", 0) or 0
            if direction == "bearish" and sentiment_score > 0:
                sentiment_score *= 0.5  # Dampen positive sentiment when market is bearish
            elif direction == "bullish" and sentiment_score < 0:
                sentiment_score *= 0.5  # Dampen negative sentiment when market is bullish

            sentiment_score = round(float(np.clip(sentiment_score, -100, 100)), 1)
            if sentiment_score > 15:
                sentiment_label = "positive"
            elif sentiment_score < -15:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

            # Random timestamp within last 2 hours
            minutes_ago = int(rng.integers(1, 120))
            ts = now - timedelta(minutes=minutes_ago)
            is_breaking = minutes_ago <= BREAKING_NEWS_WINDOW_MIN and impact == "high"

            keywords = _extract_keywords(headline)

            items.append(NewsItem(
                headline=headline,
                symbol=symbol,
                category=category,
                impact=impact,
                sentiment=sentiment_label,
                sentiment_score=sentiment_score,
                timestamp=ts.isoformat(),
                is_breaking=is_breaking,
                keywords=keywords,
            ))

    # Sort by timestamp (most recent first)
    items.sort(key=lambda n: n.timestamp, reverse=True)

    return items


# ═══════════════════════════════════════════════════════════════════════════════
# NEWS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def _aggregate_sentiment_by_class(items: list[NewsItem]) -> dict[str, float]:
    """Compute average sentiment score per asset class."""
    class_scores: dict[str, list[float]] = {}

    for item in items:
        asset_class = _classify_asset_class(item.symbol)
        if asset_class not in class_scores:
            class_scores[asset_class] = []
        class_scores[asset_class].append(item.sentiment_score)

    return {
        cls: round(float(np.mean(scores)), 1)
        for cls, scores in class_scores.items()
        if scores
    }


def _cluster_news(items: list[NewsItem]) -> list[dict]:
    """Group related news items by shared keywords.

    Uses a simple keyword-overlap approach: two items are in the same cluster
    if they share at least CLUSTER_SIMILARITY_THRESHOLD keywords.
    """
    n = len(items)
    if n == 0:
        return []

    # Build adjacency based on keyword overlap
    assigned = [-1] * n
    cluster_id = 0

    for i in range(n):
        if assigned[i] >= 0:
            continue

        # Start a new cluster with item i
        assigned[i] = cluster_id
        kw_i = set(items[i].keywords)

        if not kw_i:
            cluster_id += 1
            continue

        for j in range(i + 1, n):
            if assigned[j] >= 0:
                continue
            kw_j = set(items[j].keywords)
            overlap = len(kw_i & kw_j)
            if overlap >= CLUSTER_SIMILARITY_THRESHOLD:
                assigned[j] = cluster_id
                kw_i.update(kw_j)  # Expand cluster keywords

        cluster_id += 1

    # Build cluster summaries
    clusters: dict[int, list[int]] = {}
    for idx, cid in enumerate(assigned):
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(idx)
        items[idx].cluster_id = cid

    result = []
    for cid, member_indices in clusters.items():
        if len(member_indices) < 2:
            continue  # Skip singletons

        members = [items[i] for i in member_indices]
        # Determine cluster topic from most common keywords
        all_kw: dict[str, int] = {}
        for m in members:
            for kw in m.keywords:
                all_kw[kw] = all_kw.get(kw, 0) + 1
        top_keywords = sorted(all_kw, key=all_kw.get, reverse=True)[:3]

        avg_sent = round(float(np.mean([m.sentiment_score for m in members])), 1)

        result.append({
            "cluster_id": cid,
            "topic": ", ".join(top_keywords) if top_keywords else "general",
            "headline_count": len(members),
            "symbols": list(set(m.symbol for m in members)),
            "avg_sentiment": avg_sent,
            "top_headlines": [m.headline for m in members[:3]],
        })

    # Sort by cluster size descending
    result.sort(key=lambda c: c["headline_count"], reverse=True)

    return result


def _generate_risk_adjustments(
    items: list[NewsItem],
    aggregate_sentiment: dict[str, float],
) -> list[dict]:
    """Generate risk adjustment suggestions based on news analysis."""
    adjustments: list[dict] = []

    # Check for strongly negative asset classes
    for asset_class, avg_sent in aggregate_sentiment.items():
        if avg_sent < -30:
            adjustments.append({
                "asset_class": asset_class,
                "action": "reduce_exposure",
                "severity": "high" if avg_sent < -50 else "medium",
                "reason": f"Strongly negative news sentiment ({avg_sent:+.0f}) across {asset_class}",
                "suggested_adjustment_pct": round(min(25, abs(avg_sent) * 0.4), 1),
            })
        elif avg_sent > 40:
            adjustments.append({
                "asset_class": asset_class,
                "action": "increase_exposure",
                "severity": "low",
                "reason": f"Positive news sentiment ({avg_sent:+.0f}) supports {asset_class} positions",
                "suggested_adjustment_pct": round(min(15, avg_sent * 0.2), 1),
            })

    # Check for high-impact breaking news
    breaking_items = [i for i in items if i.is_breaking]
    if len(breaking_items) >= 2:
        negative_breaking = [i for i in breaking_items if i.sentiment_score < -20]
        if negative_breaking:
            adjustments.append({
                "asset_class": "all",
                "action": "activate_hedges",
                "severity": "high",
                "reason": f"{len(negative_breaking)} negative breaking news events detected — consider emergency hedging",
                "suggested_adjustment_pct": 10.0,
            })

    return adjustments


# ═══════════════════════════════════════════════════════════════════════════════
# AIRIA ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _enrich_with_airia(top_headlines: list[dict]) -> dict[str, Any] | None:
    """Send top headlines to Airia for AI-powered sentiment analysis (best-effort).

    Returns:
        Dict with AI sentiment analysis or None if unavailable
    """
    if not bridge.is_available:
        return None

    try:
        input_data = {
            "headlines": top_headlines[:15],
            "instruction": (
                "Analyze these financial headlines. For each, provide a sentiment score "
                "(-100 to +100), impact assessment (high/medium/low), and a brief "
                "market implications note. Return JSON with key 'analysis'."
            ),
        }

        result = bridge.execute_sentiment_scoring(top_headlines)

        if result.get("ok"):
            console.print(f"  [dim]Airia news sentiment: {result.get('latency_ms', 0)}ms[/]")
            return result.get("parsed") or result.get("result")

        console.print(f"  [yellow]Airia news sentiment failed: {result.get('error', '?')}[/]")
    except Exception as e:
        console.print(f"  [yellow]Airia news sentiment error: {e}[/]")

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT RUN — Point d'entree principal
# ═══════════════════════════════════════════════════════════════════════════════

async def run(
    run_id: str,
    signals: list[dict],
) -> StepResult:
    """Execute the News Scanner agent: generate, classify, cluster, and analyze news.

    Args:
        run_id: Pipeline run identifier
        signals: List of MarketSignal dicts (symbol, price, change_24h, volatility, direction, ...)

    Returns:
        StepResult with NewsScanReport in data
    """
    t0 = time.monotonic()
    console.print("[bold cyan]Agent Sentinel — News Scanner[/] scanning headlines...")

    if not signals:
        latency = (time.monotonic() - t0) * 1000
        console.print("[yellow]News Scanner: no signals to scan[/]")
        return StepResult(
            step_name="news_scanning",
            status=StepStatus.FAILED,
            error="No signals provided",
            agent_used="news_scanner",
            latency_ms=latency,
        )

    models_used = ["keyword_nlp"]

    # === Step 1: Generate simulated news feed ===
    seed = hash(run_id) % (2**31)
    items = _generate_news_feed(signals, seed=seed)
    console.print(f"  [dim]Generated {len(items)} headlines for {len(signals)} assets[/]")

    # === Step 2: Aggregate sentiment per asset class ===
    aggregate_sentiment = _aggregate_sentiment_by_class(items)

    # === Step 3: Cluster related news ===
    clusters = _cluster_news(items)
    console.print(f"  [dim]Found {len(clusters)} news clusters[/]")

    # === Step 4: Detect breaking news ===
    breaking_items = [i for i in items if i.is_breaking]
    high_impact_items = [i for i in items if i.impact == "high"]
    console.print(
        f"  [dim]Breaking: {len(breaking_items)}, High-impact: {len(high_impact_items)}[/]"
    )

    # === Step 5: Generate risk adjustments ===
    risk_adjustments = _generate_risk_adjustments(items, aggregate_sentiment)

    # === Step 6: Airia enrichment (best-effort) ===
    top_for_airia = [
        {
            "headline": item.headline,
            "symbol": item.symbol,
            "category": item.category,
            "impact": item.impact,
            "sentiment_score": item.sentiment_score,
        }
        for item in items[:15]
    ]

    airia_result = _enrich_with_airia(top_for_airia)
    airia_enriched = airia_result is not None
    if airia_enriched:
        models_used.append("airia")

        # Integrate Airia sentiment adjustments if available
        if isinstance(airia_result, dict):
            for analysis in airia_result.get("analysis", airia_result.get("by_asset", [])):
                if isinstance(analysis, dict):
                    symbol = analysis.get("asset", analysis.get("symbol", ""))
                    ai_score = analysis.get("sentiment", analysis.get("score"))
                    if symbol and ai_score is not None:
                        matching = [i for i in items if i.symbol == symbol]
                        for m in matching[:1]:
                            # Blend: 70% local + 30% Airia
                            blended = m.sentiment_score * 0.7 + float(ai_score) * 0.3
                            m.sentiment_score = round(float(np.clip(blended, -100, 100)), 1)

    # === Step 7: Compute global sentiment ===
    if items:
        all_scores = np.array([i.sentiment_score for i in items])
        global_sentiment = round(float(np.mean(all_scores)), 1)
    else:
        global_sentiment = 0.0

    # Build report
    report = NewsScanReport(
        items=items,
        total_headlines=len(items),
        breaking_count=len(breaking_items),
        high_impact_count=len(high_impact_items),
        aggregate_sentiment=aggregate_sentiment,
        global_sentiment=global_sentiment,
        clusters=clusters,
        risk_adjustments=risk_adjustments,
        airia_enriched=airia_enriched,
        models_used=models_used,
    )

    latency = (time.monotonic() - t0) * 1000

    # === Audit DB ===
    db.save_audit(
        run_id, "news_scanning", "news_scanner",
        input_summary=f"{len(signals)} assets, {len(items)} headlines generated",
        output_summary=(
            f"global_sent={global_sentiment:+.0f}, breaking={len(breaking_items)}, "
            f"high_impact={len(high_impact_items)}, clusters={len(clusters)}, "
            f"risk_adjustments={len(risk_adjustments)}, airia={'YES' if airia_enriched else 'NO'}"
        ),
        model_used=", ".join(models_used),
        latency_ms=latency,
    )

    # === Display ===
    _print_news_report(report)

    # Confidence: based on number of headlines and consensus in sentiment
    if items:
        sentiment_std = float(np.std(all_scores))
        # Lower std = more consensus = higher confidence
        confidence = round(min(95, max(40, 85 - sentiment_std * 0.5)), 1)
    else:
        confidence = 30.0

    console.print(
        f"[green]News Scanner done[/] — {len(items)} headlines, "
        f"global_sent={global_sentiment:+.0f}, breaking={len(breaking_items)}, "
        f"{len(clusters)} clusters in {int(latency)}ms"
    )

    return StepResult(
        step_name="news_scanning",
        status=StepStatus.SUCCESS,
        data={
            "headlines": [
                {
                    "headline": i.headline,
                    "symbol": i.symbol,
                    "category": i.category,
                    "impact": i.impact,
                    "sentiment": i.sentiment,
                    "sentiment_score": i.sentiment_score,
                    "timestamp": i.timestamp,
                    "is_breaking": i.is_breaking,
                    "cluster_id": i.cluster_id,
                }
                for i in items
            ],
            "total_headlines": len(items),
            "breaking_count": len(breaking_items),
            "high_impact_count": len(high_impact_items),
            "aggregate_sentiment": aggregate_sentiment,
            "global_sentiment": global_sentiment,
            "clusters": clusters,
            "risk_adjustments": risk_adjustments,
            "airia_enriched": airia_enriched,
            "models_used": models_used,
        },
        confidence=confidence,
        agent_used="news_scanner",
        model_used=", ".join(models_used),
        latency_ms=latency,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _sentiment_color(score: float) -> str:
    """Return Rich color string for a sentiment score."""
    if score > 30:
        return "bold green"
    if score > 10:
        return "green"
    if score > -10:
        return "yellow"
    if score > -30:
        return "red"
    return "bold red"


def _impact_color(impact: str) -> str:
    """Return Rich color string for an impact level."""
    return {"high": "bold red", "medium": "yellow", "low": "dim"}.get(impact, "white")


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def _print_news_report(report: NewsScanReport) -> None:
    """Display the complete news scanning report."""
    if not report.items:
        console.print("[yellow]No headlines to display[/]")
        return

    # === Breaking News Panel ===
    breaking = [i for i in report.items if i.is_breaking]
    if breaking:
        breaking_text = ""
        for b in breaking[:5]:
            sc = _sentiment_color(b.sentiment_score)
            breaking_text += (
                f"[bold]{b.symbol}[/] [{sc}]{b.sentiment_score:+.0f}[/] "
                f"{b.headline}\n"
            )
        console.print(Panel(
            breaking_text.strip(),
            title="[bold red]BREAKING NEWS[/]",
            border_style="red",
        ))

    # === Headlines Table ===
    table = Table(
        title=f"News Feed — {report.total_headlines} Headlines",
        show_lines=False,
    )
    table.add_column("Symbol", style="cyan", width=12)
    table.add_column("Headline", max_width=50)
    table.add_column("Cat.", width=12)
    table.add_column("Impact", width=8, justify="center")
    table.add_column("Sent.", width=8, justify="right")
    table.add_column("Brk", width=4, justify="center")

    for item in report.items[:MAX_NEWS_DISPLAY]:
        sc = _sentiment_color(item.sentiment_score)
        ic = _impact_color(item.impact)
        brk_str = "[bold red]![/]" if item.is_breaking else ""

        # Truncate headline (remove symbol prefix for display since it's in its own column)
        headline_display = item.headline
        if headline_display.startswith(f"[{item.symbol}] "):
            headline_display = headline_display[len(f"[{item.symbol}] "):]

        table.add_row(
            item.symbol,
            headline_display[:50],
            item.category,
            f"[{ic}]{item.impact}[/]",
            f"[{sc}]{item.sentiment_score:+.0f}[/]",
            brk_str,
        )

    console.print(table)

    # === Aggregate Sentiment Table ===
    if report.aggregate_sentiment:
        agg_table = Table(title="Aggregate Sentiment by Asset Class", show_lines=False)
        agg_table.add_column("Asset Class", style="cyan")
        agg_table.add_column("Avg Sentiment", justify="right")
        agg_table.add_column("Bias")

        for cls, score in sorted(report.aggregate_sentiment.items()):
            sc = _sentiment_color(score)
            bias = "Bullish" if score > 10 else ("Bearish" if score < -10 else "Neutral")
            agg_table.add_row(
                cls,
                f"[{sc}]{score:+.1f}[/]",
                f"[{sc}]{bias}[/]",
            )

        console.print(agg_table)

    # === Clusters ===
    if report.clusters:
        for cluster in report.clusters[:5]:
            sc = _sentiment_color(cluster["avg_sentiment"])
            console.print(
                f"  [dim]Cluster:[/] [cyan]{cluster['topic']}[/] — "
                f"{cluster['headline_count']} headlines, "
                f"[{sc}]sent={cluster['avg_sentiment']:+.1f}[/], "
                f"symbols: {', '.join(cluster['symbols'][:5])}"
            )

    # === Synthesis Panel ===
    sent_color = _sentiment_color(report.global_sentiment)
    adj_text = ""
    if report.risk_adjustments:
        adj_lines = []
        for adj in report.risk_adjustments[:3]:
            adj_lines.append(
                f"  [{adj['severity'].upper()}] {adj['asset_class']}: "
                f"{adj['action']} ({adj['reason'][:60]})"
            )
        adj_text = "\n[bold]Risk Adjustments:[/]\n" + "\n".join(adj_lines)

    console.print(Panel(
        f"[bold]Global News Sentiment:[/] [{sent_color}]{report.global_sentiment:+.1f}[/]\n"
        f"[bold]Total Headlines:[/] {report.total_headlines}\n"
        f"[bold red]Breaking:[/] {report.breaking_count}  "
        f"[bold yellow]High Impact:[/] {report.high_impact_count}\n"
        f"[bold]Clusters:[/] {len(report.clusters)}\n"
        f"[bold]Airia Enriched:[/] {'YES' if report.airia_enriched else 'NO'}\n"
        f"[bold]Models:[/] {', '.join(report.models_used)}"
        f"{adj_text}",
        title="[bold]News Scanner — Synthesis[/]",
        border_style=sent_color.replace("bold ", ""),
    ))

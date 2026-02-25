"""Agent Sentinel — Correlation Matrix: cross-asset correlation analysis and regime detection.

Builds a pseudo-correlation matrix from multi-timeframe price changes and performs:
- Pairwise correlation computation (change_1h, change_24h, change_7d)
- Correlation cluster detection (groups of co-moving assets)
- Regime classification: "normal" (avg corr < 0.4), "elevated" (0.4-0.7), "crisis" (> 0.7)
- Top correlated and anti-correlated pair identification
- Eigenvalue analysis for portfolio concentration risk (PCA-style)
- Contagion risk scoring based on correlation regime

Enrichissement Airia best-effort via bridge (non-bloquant).
Utilise numpy pour les operations matricielles.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
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
# CONFIGURATION — Correlation analysis parameters
# ═══════════════════════════════════════════════════════════════════════════════

# Timeframe weights for composite correlation
TIMEFRAME_WEIGHTS = {
    "change_1h":  0.20,
    "change_24h": 0.50,
    "change_7d":  0.30,
}

# Correlation regime thresholds (average absolute correlation)
REGIME_NORMAL_MAX = 0.40
REGIME_ELEVATED_MAX = 0.70
# Above REGIME_ELEVATED_MAX = crisis

# Cluster detection: correlation above this threshold groups assets together
CLUSTER_CORRELATION_THRESHOLD = 0.65

# Diversification threshold: pairs below this are diversification candidates
DIVERSIFICATION_THRESHOLD = -0.20

# Top pairs to report
TOP_CORRELATED_PAIRS = 5
TOP_DIVERSIFICATION_PAIRS = 5

# Eigenvalue concentration: first eigenvalue / sum > threshold = concentrated portfolio
EIGENVALUE_CONCENTRATION_WARNING = 0.50
EIGENVALUE_CONCENTRATION_CRITICAL = 0.70

# Minimum signals for analysis
MIN_SIGNALS = 3

# Contagion risk scoring weights
CONTAGION_REGIME_WEIGHT = 0.40
CONTAGION_CONCENTRATION_WEIGHT = 0.35
CONTAGION_CLUSTER_WEIGHT = 0.25


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CorrelationPair:
    """A correlated pair of assets."""
    asset_1: str
    asset_2: str
    correlation: float = 0.0
    pair_type: str = "positive"     # "positive", "negative", "neutral"
    timeframe_detail: dict[str, float] = field(default_factory=dict)


@dataclass
class CorrelationCluster:
    """A group of correlated assets moving together."""
    cluster_id: int = 0
    assets: list[str] = field(default_factory=list)
    avg_internal_corr: float = 0.0
    dominant_direction: str = "mixed"   # "bullish", "bearish", "mixed"


@dataclass
class EigenAnalysis:
    """Portfolio concentration analysis via eigenvalues."""
    top_eigenvalue_ratio: float = 0.0   # First eigenvalue / sum
    effective_dimensions: int = 0       # Number of eigenvalues explaining 90% variance
    concentration_level: str = "low"    # "low", "moderate", "high", "critical"
    eigenvalues: list[float] = field(default_factory=list)


@dataclass
class CorrelationReport:
    """Complete correlation analysis report."""
    correlation_matrix: list[list[float]] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    top_correlated: list[CorrelationPair] = field(default_factory=list)
    diversification_opportunities: list[CorrelationPair] = field(default_factory=list)
    clusters: list[CorrelationCluster] = field(default_factory=list)
    eigen_analysis: EigenAnalysis = field(default_factory=EigenAnalysis)
    regime: str = "normal"
    avg_correlation: float = 0.0
    contagion_risk_score: float = 0.0
    total_assets: int = 0
    airia_enriched: bool = False
    models_used: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# CORRELATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _build_feature_matrix(signals: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Build a feature matrix from multi-timeframe price changes.

    Each row is an asset, each column is a weighted timeframe change.

    Args:
        signals: List of signal dicts with change_1h, change_24h, change_7d

    Returns:
        (feature_matrix [N x T], list of symbol names)
    """
    symbols: list[str] = []
    rows: list[list[float]] = []

    for sig in signals:
        symbol = sig.get("symbol", "?")
        row = []
        for tf, weight in TIMEFRAME_WEIGHTS.items():
            value = float(sig.get(tf, 0) or 0)
            row.append(value * weight)
        symbols.append(symbol)
        rows.append(row)

    matrix = np.array(rows, dtype=np.float64)
    return matrix, symbols


def _compute_correlation_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise correlation matrix from the feature matrix.

    Uses numpy's corrcoef on the feature matrix rows (each row = one asset).
    Handles edge cases where standard deviation is zero.

    Args:
        feature_matrix: Shape [N_assets, N_features]

    Returns:
        Correlation matrix [N_assets, N_assets]
    """
    n = feature_matrix.shape[0]
    if n < 2:
        return np.eye(n)

    # Check for zero-variance rows
    stds = np.std(feature_matrix, axis=1)
    has_variance = stds > 1e-10

    # Replace zero-variance rows with small noise to avoid NaN
    cleaned = feature_matrix.copy()
    for i in range(n):
        if not has_variance[i]:
            cleaned[i] = np.random.default_rng(42).normal(0, 1e-6, feature_matrix.shape[1])

    corr = np.corrcoef(cleaned)

    # Ensure diagonal is exactly 1.0
    np.fill_diagonal(corr, 1.0)

    # Replace any NaN with 0
    corr = np.nan_to_num(corr, nan=0.0)

    return corr


def _extract_top_pairs(
    corr_matrix: np.ndarray,
    symbols: list[str],
    top_n: int,
    mode: str = "positive",
) -> list[CorrelationPair]:
    """Extract top correlated or anti-correlated pairs from the matrix.

    Args:
        corr_matrix: Symmetric correlation matrix [N x N]
        symbols: List of asset symbols
        top_n: Number of top pairs to return
        mode: "positive" for most correlated, "negative" for most anti-correlated

    Returns:
        List of CorrelationPair sorted by relevance
    """
    n = len(symbols)
    pairs: list[CorrelationPair] = []

    for i in range(n):
        for j in range(i + 1, n):
            corr_val = float(corr_matrix[i, j])
            pairs.append(CorrelationPair(
                asset_1=symbols[i],
                asset_2=symbols[j],
                correlation=round(corr_val, 4),
                pair_type="positive" if corr_val > 0 else "negative",
            ))

    if mode == "positive":
        pairs.sort(key=lambda p: p.correlation, reverse=True)
    else:
        pairs.sort(key=lambda p: p.correlation)

    return pairs[:top_n]


def _detect_clusters(
    corr_matrix: np.ndarray,
    symbols: list[str],
    signals: list[dict],
    threshold: float = CLUSTER_CORRELATION_THRESHOLD,
) -> list[CorrelationCluster]:
    """Detect correlation clusters via simple greedy grouping.

    Starting from the most correlated pair, greedily expand clusters
    by including assets whose average correlation with the cluster exceeds threshold.

    Args:
        corr_matrix: Correlation matrix [N x N]
        symbols: Asset symbols
        signals: Original signal dicts (for direction information)
        threshold: Minimum average correlation to join a cluster

    Returns:
        List of detected CorrelationCluster
    """
    n = len(symbols)
    assigned: set[int] = set()
    clusters: list[CorrelationCluster] = []
    cluster_id = 0

    # Build symbol-to-index map
    changes_map = {}
    for i, sig in enumerate(signals):
        changes_map[i] = float(sig.get("change_24h", 0) or 0)

    # Sort all pairs by correlation descending
    pair_indices: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_indices.append((i, j, float(corr_matrix[i, j])))
    pair_indices.sort(key=lambda x: x[2], reverse=True)

    for idx_a, idx_b, corr_val in pair_indices:
        if corr_val < threshold:
            break

        if idx_a in assigned and idx_b in assigned:
            continue

        # Start or extend cluster
        cluster_members: list[int] = []
        if idx_a not in assigned:
            cluster_members.append(idx_a)
            assigned.add(idx_a)
        if idx_b not in assigned:
            cluster_members.append(idx_b)
            assigned.add(idx_b)

        if not cluster_members:
            continue

        # Try to add unassigned neighbors
        for k in range(n):
            if k in assigned:
                continue
            avg_corr_with_cluster = np.mean([
                float(corr_matrix[k, m]) for m in cluster_members
            ])
            if avg_corr_with_cluster >= threshold:
                cluster_members.append(k)
                assigned.add(k)

        if len(cluster_members) < 2:
            continue

        # Compute internal metrics
        internal_corrs = []
        for i_c in range(len(cluster_members)):
            for j_c in range(i_c + 1, len(cluster_members)):
                internal_corrs.append(float(corr_matrix[cluster_members[i_c], cluster_members[j_c]]))

        avg_internal = float(np.mean(internal_corrs)) if internal_corrs else 0.0

        # Direction
        member_changes = [changes_map.get(m, 0) for m in cluster_members]
        avg_change = float(np.mean(member_changes)) if member_changes else 0.0
        if avg_change > 0.5:
            direction = "bullish"
        elif avg_change < -0.5:
            direction = "bearish"
        else:
            direction = "mixed"

        cluster_id += 1
        clusters.append(CorrelationCluster(
            cluster_id=cluster_id,
            assets=[symbols[m] for m in cluster_members],
            avg_internal_corr=round(avg_internal, 3),
            dominant_direction=direction,
        ))

    return clusters


def _eigenvalue_analysis(corr_matrix: np.ndarray) -> EigenAnalysis:
    """Perform eigenvalue analysis on the correlation matrix for concentration risk.

    The ratio of the first eigenvalue to the sum measures how much of the
    portfolio variance is driven by a single factor (systemic risk).

    Args:
        corr_matrix: Correlation matrix [N x N]

    Returns:
        EigenAnalysis with concentration metrics
    """
    n = corr_matrix.shape[0]
    if n < 2:
        return EigenAnalysis(
            top_eigenvalue_ratio=1.0,
            effective_dimensions=1,
            concentration_level="critical",
            eigenvalues=[1.0],
        )

    # Compute eigenvalues (real parts — correlation matrix is symmetric)
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
    eigenvalues = np.maximum(eigenvalues, 0)  # Remove tiny negatives from numerical error

    total = float(np.sum(eigenvalues))
    if total <= 0:
        return EigenAnalysis(
            top_eigenvalue_ratio=0.0,
            effective_dimensions=n,
            concentration_level="low",
            eigenvalues=[],
        )

    top_ratio = float(eigenvalues[0] / total)

    # Effective dimensions: number of eigenvalues explaining 90% of variance
    cumulative = np.cumsum(eigenvalues) / total
    effective_dims = int(np.searchsorted(cumulative, 0.90)) + 1
    effective_dims = min(effective_dims, n)

    # Classification
    if top_ratio >= EIGENVALUE_CONCENTRATION_CRITICAL:
        level = "critical"
    elif top_ratio >= EIGENVALUE_CONCENTRATION_WARNING:
        level = "high"
    elif top_ratio >= 0.35:
        level = "moderate"
    else:
        level = "low"

    return EigenAnalysis(
        top_eigenvalue_ratio=round(top_ratio, 4),
        effective_dimensions=effective_dims,
        concentration_level=level,
        eigenvalues=[round(float(e), 4) for e in eigenvalues[:10]],
    )


def _classify_regime(avg_abs_correlation: float) -> str:
    """Classify correlation regime based on average absolute correlation.

    Args:
        avg_abs_correlation: Average absolute pairwise correlation

    Returns:
        One of "normal", "elevated", "crisis"
    """
    if avg_abs_correlation >= REGIME_ELEVATED_MAX:
        return "crisis"
    if avg_abs_correlation >= REGIME_NORMAL_MAX:
        return "elevated"
    return "normal"


def _compute_contagion_risk(
    regime: str,
    eigen_analysis: EigenAnalysis,
    clusters: list[CorrelationCluster],
    total_assets: int,
) -> float:
    """Compute contagion risk score (0-100) from correlation indicators.

    Weighted combination of:
    - Regime severity (normal=0, elevated=50, crisis=100)
    - Eigenvalue concentration (top_ratio * 100)
    - Cluster coverage (% of assets in clusters)

    Args:
        regime: Correlation regime string
        eigen_analysis: Eigenvalue analysis results
        clusters: Detected correlation clusters
        total_assets: Total number of assets analyzed

    Returns:
        Contagion risk score 0-100
    """
    # Regime score
    regime_scores = {"normal": 10.0, "elevated": 55.0, "crisis": 95.0}
    regime_score = regime_scores.get(regime, 30.0)

    # Concentration score
    concentration_score = min(100.0, eigen_analysis.top_eigenvalue_ratio * 130.0)

    # Cluster coverage score
    clustered_assets = set()
    for c in clusters:
        clustered_assets.update(c.assets)
    cluster_coverage = len(clustered_assets) / max(total_assets, 1) * 100.0
    cluster_score = min(100.0, cluster_coverage * 1.2)

    contagion = (
        CONTAGION_REGIME_WEIGHT * regime_score
        + CONTAGION_CONCENTRATION_WEIGHT * concentration_score
        + CONTAGION_CLUSTER_WEIGHT * cluster_score
    )

    return round(float(np.clip(contagion, 0.0, 100.0)), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# AIRIA ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _enrich_with_airia(
    report: CorrelationReport,
) -> dict[str, Any] | None:
    """Enrichissement via Airia — analyse narrative des correlations (best-effort).

    Returns:
        Dict with Airia analysis or None if unavailable
    """
    if not bridge.is_available:
        return None

    try:
        input_data = {
            "correlation_summary": {
                "regime": report.regime,
                "avg_correlation": report.avg_correlation,
                "contagion_risk": report.contagion_risk_score,
                "top_pairs": [
                    {"a1": p.asset_1, "a2": p.asset_2, "corr": p.correlation}
                    for p in report.top_correlated[:5]
                ],
                "clusters": [
                    {"assets": c.assets, "direction": c.dominant_direction}
                    for c in report.clusters[:5]
                ],
                "concentration": report.eigen_analysis.concentration_level,
            },
            "instruction": "Analyze cross-asset correlations and assess contagion risks.",
        }

        result = bridge.execute_risk_aggregation(input_data)

        if result.get("ok"):
            console.print(f"  [dim]Airia correlation: {result.get('latency_ms', 0)}ms[/]")
            return result.get("parsed") or result.get("result")

        console.print(f"  [yellow]Airia correlation failed: {result.get('error', '?')}[/]")
    except Exception as e:
        console.print(f"  [yellow]Airia correlation error: {e}[/]")

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT RUN — Point d'entree principal
# ═══════════════════════════════════════════════════════════════════════════════

async def run(
    run_id: str,
    signals: list[dict],
) -> StepResult:
    """Execute l'agent Correlation Matrix: analyse cross-asset et detection regime.

    Args:
        run_id: Identifiant du run pipeline
        signals: Liste de signaux marche (dicts avec symbol, change_1h,
                 change_24h, change_7d, asset_class, ...)

    Returns:
        StepResult avec CorrelationReport dans data
    """
    t0 = time.monotonic()
    console.print("[bold cyan]Agent Sentinel — Correlation Matrix[/] analyse en cours...")

    if not signals or len(signals) < MIN_SIGNALS:
        latency = (time.monotonic() - t0) * 1000
        console.print("[yellow]Correlation Matrix: pas assez de signaux[/]")
        return StepResult(
            step_name="correlation_analysis",
            status=StepStatus.FAILED,
            error=f"Insuffisant: {len(signals or [])} signaux (min {MIN_SIGNALS})",
            agent_used="correlation_matrix",
            latency_ms=latency,
        )

    models_used = ["numpy_correlation"]

    # Normalize input
    signal_dicts: list[dict] = []
    for s in signals:
        if isinstance(s, dict):
            signal_dicts.append(s)
        elif hasattr(s, "model_dump"):
            signal_dicts.append(s.model_dump(mode="json"))
        else:
            signal_dicts.append({"symbol": str(s)})

    # === Etape 1: Build feature matrix ===
    feature_matrix, symbols = _build_feature_matrix(signal_dicts)
    n = len(symbols)
    console.print(f"  [dim]Feature matrix: {n} assets x {feature_matrix.shape[1]} features[/]")

    # === Etape 2: Compute correlation matrix ===
    corr_matrix = _compute_correlation_matrix(feature_matrix)
    console.print(f"  [dim]Correlation matrix: {n}x{n} computed[/]")

    # === Etape 3: Average absolute correlation ===
    upper_triangle = corr_matrix[np.triu_indices(n, k=1)]
    avg_abs_corr = float(np.mean(np.abs(upper_triangle))) if len(upper_triangle) > 0 else 0.0
    avg_corr = float(np.mean(upper_triangle)) if len(upper_triangle) > 0 else 0.0

    # === Etape 4: Regime detection ===
    regime = _classify_regime(avg_abs_corr)
    console.print(f"  [dim]Regime: {regime} (avg |corr|={avg_abs_corr:.3f})[/]")

    # === Etape 5: Top correlated pairs ===
    top_correlated = _extract_top_pairs(corr_matrix, symbols, TOP_CORRELATED_PAIRS, "positive")
    for p in top_correlated[:3]:
        console.print(f"    Top: {p.asset_1} <-> {p.asset_2}: {p.correlation:.3f}")

    # === Etape 6: Diversification opportunities ===
    diversification = _extract_top_pairs(corr_matrix, symbols, TOP_DIVERSIFICATION_PAIRS, "negative")

    # === Etape 7: Cluster detection ===
    clusters = _detect_clusters(corr_matrix, symbols, signal_dicts)
    console.print(f"  [dim]Clusters detectes: {len(clusters)}[/]")

    # === Etape 8: Eigenvalue analysis ===
    eigen = _eigenvalue_analysis(corr_matrix)
    console.print(
        f"  [dim]Eigenvalue concentration: {eigen.top_eigenvalue_ratio:.2%} "
        f"({eigen.concentration_level}), effective dims={eigen.effective_dimensions}[/]"
    )

    # === Etape 9: Contagion risk score ===
    contagion_risk = _compute_contagion_risk(regime, eigen, clusters, n)

    # === Etape 10: Build report ===
    report = CorrelationReport(
        correlation_matrix=[
            [round(float(corr_matrix[i, j]), 4) for j in range(n)]
            for i in range(n)
        ],
        symbols=symbols,
        top_correlated=top_correlated,
        diversification_opportunities=diversification,
        clusters=clusters,
        eigen_analysis=eigen,
        regime=regime,
        avg_correlation=round(avg_corr, 4),
        contagion_risk_score=contagion_risk,
        total_assets=n,
        models_used=models_used,
    )

    # === Etape 11: Airia enrichment (best-effort) ===
    airia_result = _enrich_with_airia(report)
    airia_enriched = airia_result is not None
    if airia_enriched:
        models_used.append("airia")
    report.airia_enriched = airia_enriched

    latency = (time.monotonic() - t0) * 1000

    # === Audit DB ===
    db.save_audit(
        run_id, "correlation_analysis", "correlation_matrix",
        input_summary=f"{n} assets, {len(TIMEFRAME_WEIGHTS)} timeframes",
        output_summary=(
            f"regime={regime}, avg_corr={avg_corr:.3f}, contagion={contagion_risk:.0f}, "
            f"clusters={len(clusters)}, concentration={eigen.concentration_level}, "
            f"airia={'OUI' if airia_enriched else 'NON'}"
        ),
        model_used=", ".join(models_used),
        latency_ms=latency,
    )

    # === Affichage ===
    _print_correlation_report(report)

    # Confidence: higher when regime is normal, lower in crisis
    regime_confidence = {"normal": 85.0, "elevated": 65.0, "crisis": 45.0}
    confidence = regime_confidence.get(regime, 60.0)

    console.print(
        f"[green]Correlation Matrix done[/] — regime={regime}, "
        f"avg_corr={avg_corr:.3f}, contagion={contagion_risk:.0f}, "
        f"{n} assets, {len(clusters)} clusters in {int(latency)}ms"
    )

    return StepResult(
        step_name="correlation_analysis",
        status=StepStatus.SUCCESS,
        data={
            "correlation_matrix": report.correlation_matrix,
            "symbols": symbols,
            "top_correlated_pairs": [
                {
                    "asset_1": p.asset_1,
                    "asset_2": p.asset_2,
                    "correlation": p.correlation,
                    "pair_type": p.pair_type,
                }
                for p in top_correlated
            ],
            "diversification_opportunities": [
                {
                    "asset_1": p.asset_1,
                    "asset_2": p.asset_2,
                    "correlation": p.correlation,
                }
                for p in diversification
            ],
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "assets": c.assets,
                    "avg_internal_corr": c.avg_internal_corr,
                    "dominant_direction": c.dominant_direction,
                }
                for c in clusters
            ],
            "eigen_analysis": {
                "top_eigenvalue_ratio": eigen.top_eigenvalue_ratio,
                "effective_dimensions": eigen.effective_dimensions,
                "concentration_level": eigen.concentration_level,
                "eigenvalues_top10": eigen.eigenvalues,
            },
            "regime": regime,
            "avg_correlation": report.avg_correlation,
            "contagion_risk_score": contagion_risk,
            "total_assets": n,
            "airia_enriched": airia_enriched,
            "models_used": models_used,
        },
        confidence=round(confidence, 1),
        agent_used="correlation_matrix",
        model_used=", ".join(models_used),
        latency_ms=latency,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _regime_color(regime: str) -> str:
    """Return Rich color string for a correlation regime."""
    return {"normal": "green", "elevated": "yellow", "crisis": "bold red"}.get(regime, "white")


def _corr_color(corr: float) -> str:
    """Return Rich color for a correlation value."""
    abs_corr = abs(corr)
    if abs_corr >= 0.7:
        return "bold red" if corr > 0 else "bold cyan"
    if abs_corr >= 0.4:
        return "yellow"
    return "green"


def _concentration_color(level: str) -> str:
    """Return Rich color for concentration level."""
    return {
        "low": "green",
        "moderate": "yellow",
        "high": "red",
        "critical": "bold red",
    }.get(level, "white")


# ═══════════════════════════════════════════════════════════════════════════════
# AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _print_correlation_report(report: CorrelationReport) -> None:
    """Affiche le rapport de correlation complet."""

    # === Table: Top Correlated Pairs ===
    if report.top_correlated:
        table = Table(
            title=f"Top {len(report.top_correlated)} Correlated Pairs",
            show_lines=False,
        )
        table.add_column("#", style="dim", justify="right")
        table.add_column("Asset 1", style="cyan")
        table.add_column("Asset 2", style="cyan")
        table.add_column("Correlation", justify="right")
        table.add_column("Type")

        for i, p in enumerate(report.top_correlated, 1):
            color = _corr_color(p.correlation)
            table.add_row(
                str(i),
                p.asset_1,
                p.asset_2,
                f"[{color}]{p.correlation:+.3f}[/]",
                f"[{color}]{p.pair_type}[/]",
            )

        console.print(table)

    # === Table: Diversification Opportunities ===
    neg_pairs = [p for p in report.diversification_opportunities if p.correlation < DIVERSIFICATION_THRESHOLD]
    if neg_pairs:
        div_table = Table(title="Diversification Opportunities (Negative Correlation)", show_lines=False)
        div_table.add_column("Asset 1", style="cyan")
        div_table.add_column("Asset 2", style="cyan")
        div_table.add_column("Correlation", justify="right")

        for p in neg_pairs[:5]:
            div_table.add_row(
                p.asset_1,
                p.asset_2,
                f"[bold cyan]{p.correlation:+.3f}[/]",
            )

        console.print(div_table)

    # === Table: Clusters ===
    if report.clusters:
        cluster_table = Table(title=f"Correlation Clusters ({len(report.clusters)})", show_lines=False)
        cluster_table.add_column("Cluster", style="dim", justify="right")
        cluster_table.add_column("Assets", style="cyan", max_width=50)
        cluster_table.add_column("Avg Corr", justify="right")
        cluster_table.add_column("Direction")

        dir_colors = {"bullish": "green", "bearish": "red", "mixed": "yellow"}

        for c in report.clusters[:10]:
            d_color = dir_colors.get(c.dominant_direction, "white")
            assets_str = ", ".join(c.assets[:6])
            if len(c.assets) > 6:
                assets_str += f" +{len(c.assets) - 6}"
            cluster_table.add_row(
                str(c.cluster_id),
                assets_str,
                f"{c.avg_internal_corr:.3f}",
                f"[{d_color}]{c.dominant_direction}[/]",
            )

        console.print(cluster_table)

    # === Panel: Regime + Eigenvalue + Contagion ===
    r_color = _regime_color(report.regime)
    c_color = _concentration_color(report.eigen_analysis.concentration_level)
    contagion_color = "green" if report.contagion_risk_score < 30 else (
        "yellow" if report.contagion_risk_score < 60 else "red"
    )

    console.print(Panel(
        f"[bold]Correlation Regime:[/] [{r_color}]{report.regime.upper()}[/] "
        f"(avg |corr|={abs(report.avg_correlation):.3f})\n"
        f"[bold]Contagion Risk:[/] [{contagion_color}]{report.contagion_risk_score:.0f}/100[/]\n"
        f"[bold]Portfolio Concentration:[/] [{c_color}]{report.eigen_analysis.concentration_level.upper()}[/] "
        f"(top eigenvalue={report.eigen_analysis.top_eigenvalue_ratio:.1%})\n"
        f"[bold]Effective Dimensions:[/] {report.eigen_analysis.effective_dimensions} / {report.total_assets}\n"
        f"[bold]Clusters:[/] {len(report.clusters)}\n"
        f"[bold]Assets Analyzed:[/] {report.total_assets}\n"
        f"[bold]Airia Enriched:[/] {'OUI' if report.airia_enriched else 'NON'}\n"
        f"[bold]Modeles:[/] {', '.join(report.models_used)}",
        title="[bold]Correlation Matrix — Synthese[/]",
        border_style=r_color.replace("bold ", ""),
    ))

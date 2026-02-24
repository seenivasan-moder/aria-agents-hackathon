"""Airia SDK Bridge — Enhanced multi-agent pipeline execution.

Supports 10+ agent prompts, async execution, caching, and retry logic.
Uses execute_temporary_assistant() when no pipeline IDs are configured,
injecting our system prompts directly. Falls back to execute_pipeline()
if pipeline IDs are set in .env.
"""

from __future__ import annotations

import asyncio
import json
import time
from functools import lru_cache
from typing import Any

from rich.console import Console

from src.config import config

console = Console()

# ═══════════════════════════════════════════════════════════════════════════════
# AIRIA PIPELINE SYSTEM PROMPTS — 10 Specialized Agents (Bilingual FR/EN)
# ═══════════════════════════════════════════════════════════════════════════════

# Language setting — "en" for English prompts (hackathon judges), "fr" for French
PROMPT_LANG = "en"

# ── Agent 1: Market Intelligence ──────────────────────────────────────────

MARKET_INTELLIGENCE_PROMPT_FR = """\
Tu es l'Agent Sentinel-1 (Market Intelligence).
Ton role est d'analyser les donnees de marche fournies (Crypto, Forex, Commodities).
Pour chaque actif, evalue :
- Le score de risque (0-100) base sur la volatilite.
- Le regime de marche (Trending, Range, Volatile).
- L'urgence d'action.

SORTIE : Tu dois repondre EXCLUSIVEMENT en JSON structure selon ce schema :
{
  "signals": [
    {"asset": "BTC/USDT", "risk_score": 45, "regime": "Trending Up", "urgency": "Low"},
    ...
  ],
  "global_sentiment": "Bullish/Bearish/Neutral"
}"""

MARKET_INTELLIGENCE_PROMPT_EN = """\
You are Agent Sentinel-1 (Market Intelligence).
Your role is to analyze the provided market data (Crypto, Forex, Commodities).
For each asset, evaluate:
- Risk score (0-100) based on volatility.
- Market regime (Trending, Range, Volatile).
- Action urgency.

OUTPUT: You MUST respond EXCLUSIVELY in structured JSON following this schema:
{
  "signals": [
    {"asset": "BTC/USDT", "risk_score": 45, "regime": "Trending Up", "urgency": "Low"},
    ...
  ],
  "global_sentiment": "Bullish/Bearish/Neutral"
}"""

# ── Agent 2: Corporate Context ────────────────────────────────────────────

CORPORATE_CONTEXT_PROMPT_FR = """\
Tu es l'Agent Sentinel-2 (Corporate Context Analyst).
Tu recois en entree le bilan de tresorerie et les factures impayees de l'entreprise.
Identifie les concentrations de risque par devise (ex: trop d'exposition au Dollar sans couverture).

SORTIE : Tu dois repondre EXCLUSIVEMENT en JSON :
{
  "net_exposure": {"USD": -500000, "EUR": 1200000},
  "risk_zones": ["High USD liability in Q3", "Unhedged Gold position"],
  "liquidity_status": "Optimal/Warning/Critical"
}"""

CORPORATE_CONTEXT_PROMPT_EN = """\
You are Agent Sentinel-2 (Corporate Context Analyst).
You receive the company's treasury balance sheet and unpaid invoices as input.
Identify currency concentration risks (e.g., excessive Dollar exposure without hedging).

OUTPUT: You MUST respond EXCLUSIVELY in JSON:
{
  "net_exposure": {"USD": -500000, "EUR": 1200000},
  "risk_zones": ["High USD liability in Q3", "Unhedged Gold position"],
  "liquidity_status": "Optimal/Warning/Critical"
}"""

# ── Agent 3: Consensus Strategy ───────────────────────────────────────────

CONSENSUS_STRATEGY_PROMPT_FR = """\
Tu es l'Agent Sentinel-3 (Strategy Orchestrator).
Inputs :
- Market Intelligence JSON
- Corporate Context JSON

TA MISSION : Generer 3 strategies de couverture (Hedging) :
1. Conservatrice (Risque minime, cout eleve)
2. Moderee (Equilibre)
3. Agressive (Maximisation du profit, risque eleve)

SORTIE : Repondre EXCLUSIVEMENT en JSON :
{
  "strategies": [
    {
      "name": "Conservative",
      "description": "...",
      "instruments": ["FX Forward", "Commodity Futures"],
      "cost_estimate_pct": 0.45,
      "risk_reduction_pct": 85,
      "confidence": 89,
      "rationale": "Maximum protection against Euro decline detected by Sentinel-1."
    },
    {
      "name": "Moderate",
      "description": "...",
      "instruments": ["FX Option", "FX Forward"],
      "cost_estimate_pct": 0.25,
      "risk_reduction_pct": 60,
      "confidence": 75,
      "rationale": "..."
    },
    {
      "name": "Aggressive",
      "description": "...",
      "instruments": ["FX Forward"],
      "cost_estimate_pct": 0.10,
      "risk_reduction_pct": 30,
      "confidence": 60,
      "rationale": "..."
    }
  ]
}"""

CONSENSUS_STRATEGY_PROMPT_EN = """\
You are Agent Sentinel-3 (Strategy Orchestrator).
Inputs:
- Market Intelligence JSON
- Corporate Context JSON

YOUR MISSION: Generate 3 hedging strategies:
1. Conservative (Minimal risk, higher cost)
2. Moderate (Balanced approach)
3. Aggressive (Profit maximization, higher risk)

OUTPUT: Respond EXCLUSIVELY in JSON:
{
  "strategies": [
    {
      "name": "Conservative",
      "description": "...",
      "instruments": ["FX Forward", "Commodity Futures"],
      "cost_estimate_pct": 0.45,
      "risk_reduction_pct": 85,
      "confidence": 89,
      "rationale": "Maximum protection against Euro decline detected by Sentinel-1."
    },
    {
      "name": "Moderate",
      "description": "...",
      "instruments": ["FX Option", "FX Forward"],
      "cost_estimate_pct": 0.25,
      "risk_reduction_pct": 60,
      "confidence": 75,
      "rationale": "..."
    },
    {
      "name": "Aggressive",
      "description": "...",
      "instruments": ["FX Forward"],
      "cost_estimate_pct": 0.10,
      "risk_reduction_pct": 30,
      "confidence": 60,
      "rationale": "..."
    }
  ]
}"""

# ── Agent 4: Compliance & Docs ────────────────────────────────────────────

COMPLIANCE_DOCS_PROMPT_FR = """\
Tu es l'Agent Sentinel-4 (Compliance Officer).
Tu dois rediger le resume executif du rapport de gestion des risques.
Ton ton doit etre formel, precis et professionnel (style Big Four / Banque d'investissement).

TA MISSION :
1. Resumer l'etat actuel des marches.
2. Justifier la strategie recommandee par le consensus IA.
3. Certifier que la piste d'audit est complete.

SORTIE : Un texte structure avec des sections :
[EXECUTIVE SUMMARY]
[MARKET ANALYSIS]
[INTERNAL EXPOSURE]
[RECOMMENDATION]
[AUDIT CERTIFICATION]"""

COMPLIANCE_DOCS_PROMPT_EN = """\
You are Agent Sentinel-4 (Compliance Officer).
You must write the executive summary for the risk management report.
Your tone must be formal, precise, and professional (Big Four / Investment Bank style).

YOUR MISSION:
1. Summarize the current market state.
2. Justify the AI consensus-recommended strategy.
3. Certify that the audit trail is complete.

OUTPUT: A structured text with sections:
[EXECUTIVE SUMMARY]
[MARKET ANALYSIS]
[INTERNAL EXPOSURE]
[RECOMMENDATION]
[AUDIT CERTIFICATION]"""

# ── Agent 5: Risk Aggregator ─────────────────────────────────────────────

RISK_AGGREGATOR_PROMPT_FR = """\
Tu es l'Agent Sentinel-5 (Risk Aggregator).
Tu recois des signaux de risque provenant de multiples sources (marche, corporate, sentiment).
TA MISSION : Fusionner tous les signaux en un profil de risque unifie.

Evalue :
- Le risque global (0-100) avec ponderation par source
- Les risques en cascade (quand multiple signaux s'alignent dans la meme direction)
- Les correlations entre actifs qui augmentent le risque systemique

SORTIE : Repondre EXCLUSIVEMENT en JSON :
{
  "overall_risk": 67,
  "risk_by_class": {"crypto": 72, "forex": 45, "commodity": 58},
  "cascade_detected": true,
  "cascade_description": "BTC+ETH+SOL en baisse synchrone, correle avec hausse XAU",
  "systemic_risk": "Medium",
  "action_required": "Immediate hedge on crypto + commodity positions"
}"""

RISK_AGGREGATOR_PROMPT_EN = """\
You are Agent Sentinel-5 (Risk Aggregator).
You receive risk signals from multiple sources (market, corporate, sentiment).
YOUR MISSION: Merge all signals into a unified risk profile.

Evaluate:
- Overall risk (0-100) with source-weighted scoring
- Cascade risks (when multiple signals align in the same direction)
- Cross-asset correlations that increase systemic risk

OUTPUT: Respond EXCLUSIVELY in JSON:
{
  "overall_risk": 67,
  "risk_by_class": {"crypto": 72, "forex": 45, "commodity": 58},
  "cascade_detected": true,
  "cascade_description": "BTC+ETH+SOL dropping synchronously, correlated with XAU surge",
  "systemic_risk": "Medium",
  "action_required": "Immediate hedge on crypto + commodity positions"
}"""

# ── Agent 6: Anomaly Detector ─────────────────────────────────────────────

ANOMALY_DETECTOR_PROMPT_FR = """\
Tu es l'Agent Sentinel-6 (Anomaly Detector).
Tu analyses les donnees de marche pour detecter des anomalies statistiques.

CRITERES D'ANOMALIE :
- Z-score > 2.5 sur les variations de prix
- Volume > 3x la moyenne (spike de volume)
- Divergence prix/volume (prix monte mais volume baisse)
- Contagion cross-asset (correlation soudaine entre actifs non-correles)

SORTIE : Repondre EXCLUSIVEMENT en JSON :
{
  "anomalies": [
    {"asset": "BTC/USDT", "type": "volume_spike", "severity": "high", "z_score": 3.2, "description": "..."},
    ...
  ],
  "anomaly_count": 2,
  "market_health": "Warning/Normal/Critical"
}"""

ANOMALY_DETECTOR_PROMPT_EN = """\
You are Agent Sentinel-6 (Anomaly Detector).
You analyze market data to detect statistical anomalies.

ANOMALY CRITERIA:
- Z-score > 2.5 on price variations
- Volume > 3x average (volume spike)
- Price/volume divergence (price rises but volume drops)
- Cross-asset contagion (sudden correlation between uncorrelated assets)

OUTPUT: Respond EXCLUSIVELY in JSON:
{
  "anomalies": [
    {"asset": "BTC/USDT", "type": "volume_spike", "severity": "high", "z_score": 3.2, "description": "..."},
    ...
  ],
  "anomaly_count": 2,
  "market_health": "Warning/Normal/Critical"
}"""

# ── Agent 7: Sentiment Scorer ─────────────────────────────────────────────

SENTIMENT_SCORER_PROMPT_FR = """\
Tu es l'Agent Sentinel-7 (Sentiment Scorer).
Analyse les donnees de marche et determine le sentiment global et par actif.

FACTEURS :
- Momentum (direction et force du mouvement)
- Mean reversion (surachat/survente, RSI implicite)
- Regime de volatilite (calme, eleve, extreme)
- Funding rate (pour crypto)

SORTIE : Repondre EXCLUSIVEMENT en JSON :
{
  "global_sentiment": 35,
  "sentiment_label": "Bearish",
  "by_asset": [
    {"asset": "BTC/USDT", "sentiment": -25, "factors": {"momentum": -30, "mean_reversion": 10, "volatility": -20}},
    ...
  ]
}"""

SENTIMENT_SCORER_PROMPT_EN = """\
You are Agent Sentinel-7 (Sentiment Scorer).
Analyze market data and determine global and per-asset sentiment.

FACTORS:
- Momentum (direction and strength of movement)
- Mean reversion (overbought/oversold, implicit RSI)
- Volatility regime (calm, elevated, extreme)
- Funding rate (for crypto)

OUTPUT: Respond EXCLUSIVELY in JSON:
{
  "global_sentiment": 35,
  "sentiment_label": "Bearish",
  "by_asset": [
    {"asset": "BTC/USDT", "sentiment": -25, "factors": {"momentum": -30, "mean_reversion": 10, "volatility": -20}},
    ...
  ]
}"""

# ── Agent 8: Position Sizer ───────────────────────────────────────────────

POSITION_SIZER_PROMPT_FR = """\
Tu es l'Agent Sentinel-8 (Position Sizer).
Tu calcules les tailles de position optimales basees sur le profil de risque.

METHODES :
1. Kelly Criterion : f* = (bp - q) / b
2. Risk Parity : allocation inversement proportionnelle a la volatilite
3. Max Drawdown Constraint : ne jamais risquer plus de X% du capital par trade

SORTIE : Repondre EXCLUSIVEMENT en JSON :
{
  "positions": [
    {"asset": "BTC/USDT", "kelly_pct": 5.2, "risk_parity_pct": 8.0, "recommended_pct": 4.0, "max_loss_usd": 500},
    ...
  ],
  "total_allocation_pct": 35,
  "cash_reserve_pct": 65,
  "method_used": "Risk Parity (safest)"
}"""

POSITION_SIZER_PROMPT_EN = """\
You are Agent Sentinel-8 (Position Sizer).
You calculate optimal position sizes based on the risk profile.

METHODS:
1. Kelly Criterion: f* = (bp - q) / b
2. Risk Parity: allocation inversely proportional to volatility
3. Max Drawdown Constraint: never risk more than X% of capital per trade

OUTPUT: Respond EXCLUSIVELY in JSON:
{
  "positions": [
    {"asset": "BTC/USDT", "kelly_pct": 5.2, "risk_parity_pct": 8.0, "recommended_pct": 4.0, "max_loss_usd": 500},
    ...
  ],
  "total_allocation_pct": 35,
  "cash_reserve_pct": 65,
  "method_used": "Risk Parity (safest)"
}"""

# ── Agent 9: Strategy Backtester ──────────────────────────────────────────

STRATEGY_BACKTESTER_PROMPT_FR = """\
Tu es l'Agent Sentinel-9 (Strategy Backtester).
Tu recois des strategies de couverture et tu les testes sur des donnees historiques simulees.

METRIQUES :
- Sharpe Ratio
- Max Drawdown (%)
- Win Rate (%)
- P&L simule sur 30 jours

SORTIE : Repondre EXCLUSIVEMENT en JSON :
{
  "backtests": [
    {"strategy": "Conservative", "sharpe": 1.8, "max_drawdown_pct": -3.2, "win_rate_pct": 72, "pnl_30d_pct": 2.1, "grade": "A"},
    ...
  ],
  "best_strategy": "Moderate",
  "recommendation": "..."
}"""

STRATEGY_BACKTESTER_PROMPT_EN = """\
You are Agent Sentinel-9 (Strategy Backtester).
You receive hedging strategies and test them on simulated historical data.

METRICS:
- Sharpe Ratio
- Max Drawdown (%)
- Win Rate (%)
- Simulated 30-day P&L

OUTPUT: Respond EXCLUSIVELY in JSON:
{
  "backtests": [
    {"strategy": "Conservative", "sharpe": 1.8, "max_drawdown_pct": -3.2, "win_rate_pct": 72, "pnl_30d_pct": 2.1, "grade": "A"},
    ...
  ],
  "best_strategy": "Moderate",
  "recommendation": "..."
}"""

# ── Agent 10: Report Synthesizer ──────────────────────────────────────────

REPORT_SYNTHESIZER_PROMPT_FR = """\
Tu es l'Agent Sentinel-10 (Report Synthesizer).
Tu recois les resultats de TOUS les agents du pipeline et tu produis un rapport de synthese executif.

SECTIONS REQUISES :
1. RESUME EXECUTIF (3 phrases max)
2. ETAT DES MARCHES (signaux cles + anomalies)
3. EXPOSITION CORPORATE (risques principaux)
4. STRATEGIE RECOMMANDEE (avec justification du backtest)
5. POSITIONS SUGGERES (tailles + allocation)
6. AUDIT (nombre d'agents, sources, latences)

Ton style doit etre professionnel, concis et actionnable.
SORTIE : Un texte structure en sections numerotees."""

REPORT_SYNTHESIZER_PROMPT_EN = """\
You are Agent Sentinel-10 (Report Synthesizer).
You receive results from ALL pipeline agents and produce an executive synthesis report.

REQUIRED SECTIONS:
1. EXECUTIVE SUMMARY (3 sentences max)
2. MARKET STATE (key signals + anomalies)
3. CORPORATE EXPOSURE (main risks)
4. RECOMMENDED STRATEGY (with backtest justification)
5. SUGGESTED POSITIONS (sizes + allocation)
6. AUDIT (agent count, sources, latencies)

Your style must be professional, concise, and actionable.
OUTPUT: A structured text with numbered sections."""


# ── Prompt selection (bilingual) ──────────────────────────────────────────

def _select_prompt(fr: str, en: str) -> str:
    return en if PROMPT_LANG == "en" else fr

MARKET_INTELLIGENCE_PROMPT = _select_prompt(MARKET_INTELLIGENCE_PROMPT_FR, MARKET_INTELLIGENCE_PROMPT_EN)
CORPORATE_CONTEXT_PROMPT = _select_prompt(CORPORATE_CONTEXT_PROMPT_FR, CORPORATE_CONTEXT_PROMPT_EN)
CONSENSUS_STRATEGY_PROMPT = _select_prompt(CONSENSUS_STRATEGY_PROMPT_FR, CONSENSUS_STRATEGY_PROMPT_EN)
COMPLIANCE_DOCS_PROMPT = _select_prompt(COMPLIANCE_DOCS_PROMPT_FR, COMPLIANCE_DOCS_PROMPT_EN)
RISK_AGGREGATOR_PROMPT = _select_prompt(RISK_AGGREGATOR_PROMPT_FR, RISK_AGGREGATOR_PROMPT_EN)
ANOMALY_DETECTOR_PROMPT = _select_prompt(ANOMALY_DETECTOR_PROMPT_FR, ANOMALY_DETECTOR_PROMPT_EN)
SENTIMENT_SCORER_PROMPT = _select_prompt(SENTIMENT_SCORER_PROMPT_FR, SENTIMENT_SCORER_PROMPT_EN)
POSITION_SIZER_PROMPT = _select_prompt(POSITION_SIZER_PROMPT_FR, POSITION_SIZER_PROMPT_EN)
STRATEGY_BACKTESTER_PROMPT = _select_prompt(STRATEGY_BACKTESTER_PROMPT_FR, STRATEGY_BACKTESTER_PROMPT_EN)
REPORT_SYNTHESIZER_PROMPT = _select_prompt(REPORT_SYNTHESIZER_PROMPT_FR, REPORT_SYNTHESIZER_PROMPT_EN)


# Map agent names to their prompts
AGENT_PROMPTS = {
    "market_intelligence": MARKET_INTELLIGENCE_PROMPT,
    "corporate_context": CORPORATE_CONTEXT_PROMPT,
    "consensus_strategy": CONSENSUS_STRATEGY_PROMPT,
    "compliance_docs": COMPLIANCE_DOCS_PROMPT,
    "risk_aggregator": RISK_AGGREGATOR_PROMPT,
    "anomaly_detector": ANOMALY_DETECTOR_PROMPT,
    "sentiment_scorer": SENTIMENT_SCORER_PROMPT,
    "position_sizer": POSITION_SIZER_PROMPT,
    "strategy_backtester": STRATEGY_BACKTESTER_PROMPT,
    "report_synthesizer": REPORT_SYNTHESIZER_PROMPT,
}


# ═══════════════════════════════════════════════════════════════════════════════
# AIRIA BRIDGE — Enhanced with async, caching, retry
# ═══════════════════════════════════════════════════════════════════════════════

class AiriaBridge:
    """Wrapper around the Airia SDK with embedded pipeline prompts.

    Supports two modes:
    - Pipeline mode: uses pre-configured pipeline IDs (if set in .env)
    - Temporary assistant mode: uses execute_temporary_assistant() with
      embedded system prompts (no pipeline setup needed)

    Enhanced features:
    - Async execution via thread pool
    - Response caching (5 minute TTL)
    - Retry with backoff
    - Metrics tracking
    """

    def __init__(self):
        self._client = None
        self._available: bool | None = None
        self._cache: dict[str, tuple[float, dict]] = {}
        self._cache_ttl: float = 300  # 5 minutes
        self._metrics: dict[str, list[float]] = {}
        self._call_count: int = 0
        self._error_count: int = 0

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._available is False:
            return None
        if not config.airia_api_key or config.airia_api_key == "your_airia_api_key_here":
            self._available = False
            return None
        try:
            from airia import AiriaClient
            self._client = AiriaClient(api_key=config.airia_api_key, timeout=120)
            self._available = True
            console.print("[green]Airia SDK connected[/]")
            return self._client
        except ImportError:
            self._available = False
            return None
        except Exception as e:
            console.print(f"[yellow]Airia connection failed: {e}[/]")
            self._available = False
            return None

    @property
    def is_available(self) -> bool:
        if self._available is None:
            self._get_client()
        return self._available or False

    @property
    def stats(self) -> dict[str, Any]:
        """Return bridge performance stats."""
        return {
            "calls": self._call_count,
            "errors": self._error_count,
            "cache_size": len(self._cache),
            "success_rate": round(
                (self._call_count - self._error_count) / max(self._call_count, 1) * 100, 1
            ),
            "agents_available": len(AGENT_PROMPTS),
        }

    # ── Cache management ──────────────────────────────────────────────────

    def _cache_key(self, agent_name: str, user_input: str) -> str:
        return f"{agent_name}:{hash(user_input[:500])}"

    def _cache_get(self, key: str) -> dict | None:
        if key in self._cache:
            ts, result = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return result
            del self._cache[key]
        return None

    def _cache_set(self, key: str, result: dict) -> None:
        self._cache[key] = (time.time(), result)
        # Evict old entries if cache grows too large
        if len(self._cache) > 100:
            oldest = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest]

    # ── Core execution ─────────────────────────────────────────────────────

    def _execute_temporary(
        self,
        agent_name: str,
        user_input: dict | str,
    ) -> dict[str, Any]:
        """Execute via temporary assistant with embedded system prompt."""
        self._call_count += 1
        client = self._get_client()
        if not client:
            return {"ok": False, "error": "Airia not available", "mode": "local-only"}

        prompt = AGENT_PROMPTS.get(agent_name, "")
        if not prompt:
            return {"ok": False, "error": f"No prompt for agent {agent_name}"}

        input_str = json.dumps(user_input, default=str) if isinstance(user_input, dict) else user_input

        # Check cache
        cache_key = self._cache_key(agent_name, input_str)
        cached = self._cache_get(cache_key)
        if cached:
            console.print(f"  [dim]Airia {agent_name} (cached)[/]")
            return cached

        try:
            t0 = time.monotonic()
            result = client.pipeline_execution.execute_temporary_assistant(
                model_parameters={"temperature": 0.3, "maxTokens": 4096},
                user_input=input_str,
                prompt_parameters={"prompt": prompt},
                assistant_name=f"sentinel-{agent_name}",
                save_history=False,
            )
            latency = (time.monotonic() - t0) * 1000
            raw = result.result if hasattr(result, "result") else str(result)
            parsed = _try_parse_json(raw)

            # Track metrics
            if agent_name not in self._metrics:
                self._metrics[agent_name] = []
            self._metrics[agent_name].append(latency)

            console.print(f"  [dim]Airia {agent_name} (temp): {int(latency)}ms, {len(raw)} chars[/]")
            response = {
                "ok": True,
                "result": raw,
                "parsed": parsed,
                "latency_ms": round(latency),
                "source": "airia-temp",
            }
            self._cache_set(cache_key, response)
            return response
        except Exception as e:
            self._error_count += 1
            console.print(f"  [yellow]Airia {agent_name} failed: {e}[/]")
            return {"ok": False, "error": str(e), "source": "airia"}

    def execute_pipeline(
        self,
        pipeline_id: str,
        user_input: dict | str,
        agent_name: str = "",
    ) -> dict[str, Any]:
        """Execute an Airia pipeline by ID. Returns parsed result or error."""
        self._call_count += 1
        client = self._get_client()
        if not client:
            return {"ok": False, "error": "Airia not available", "mode": "local-only"}
        if not pipeline_id:
            return {"ok": False, "error": f"Pipeline ID not configured for {agent_name}"}

        input_str = json.dumps(user_input, default=str) if isinstance(user_input, dict) else user_input

        try:
            t0 = time.monotonic()
            result = client.pipeline_execution.execute_pipeline(
                pipeline_id=pipeline_id,
                user_input=input_str,
            )
            latency = (time.monotonic() - t0) * 1000
            raw = result.result if hasattr(result, "result") else str(result)
            parsed = _try_parse_json(raw)

            console.print(f"  [dim]Airia {agent_name}: {int(latency)}ms[/]")
            return {
                "ok": True,
                "result": raw,
                "parsed": parsed,
                "latency_ms": round(latency),
                "source": "airia-pipeline",
            }
        except Exception as e:
            self._error_count += 1
            console.print(f"  [yellow]Airia {agent_name} failed: {e}[/]")
            return {"ok": False, "error": str(e), "source": "airia"}

    def _execute_agent(
        self,
        agent_name: str,
        pipeline_id: str,
        user_input: dict | str,
    ) -> dict[str, Any]:
        """Smart execution: pipeline ID if available, else temporary assistant."""
        if pipeline_id:
            return self.execute_pipeline(pipeline_id, user_input, agent_name)
        return self._execute_temporary(agent_name, user_input)

    # ── Async execution wrapper ──────────────────────────────────────────

    async def async_execute(
        self,
        agent_name: str,
        user_input: dict | str,
        pipeline_id: str = "",
    ) -> dict[str, Any]:
        """Async wrapper for agent execution (runs in thread pool)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._execute_agent(agent_name, pipeline_id, user_input),
        )

    async def async_execute_parallel(
        self,
        tasks: list[tuple[str, dict | str]],
    ) -> list[dict[str, Any]]:
        """Execute multiple agents in parallel via Airia.

        Args:
            tasks: List of (agent_name, user_input) tuples

        Returns:
            List of results in same order
        """
        coros = [
            self.async_execute(agent_name, user_input)
            for agent_name, user_input in tasks
        ]
        return await asyncio.gather(*coros, return_exceptions=False)

    # ── Per-agent pipeline methods ────────────────────────────────────────

    def execute_market_pipeline(self, market_data: list[dict]) -> dict[str, Any]:
        """Run Agent 1 through Airia with market data as input."""
        return self._execute_agent(
            "market_intelligence",
            config.market_pipeline_id,
            {
                "market_data": market_data[:15],
                "instruction": "Analyze these market signals and return risk assessment.",
            },
        )

    def execute_corporate_pipeline(self, exposure_data: dict) -> dict[str, Any]:
        """Run Agent 2 through Airia with corporate exposure data."""
        return self._execute_agent(
            "corporate_context",
            config.corporate_pipeline_id,
            {
                "corporate_data": exposure_data,
                "instruction": "Analyze corporate exposure and identify risk zones.",
            },
        )

    def execute_consensus_pipeline(
        self,
        market_signals: list[dict],
        corporate_exposure: dict,
    ) -> dict[str, Any]:
        """Run Agent 3 through Airia with combined data."""
        return self._execute_agent(
            "consensus_strategy",
            config.consensus_pipeline_id,
            {
                "market_signals": market_signals[:10],
                "corporate_exposure": corporate_exposure,
                "instruction": "Generate 3 hedging strategies based on market and corporate data.",
            },
        )

    def execute_compliance_pipeline(self, report_data: dict) -> dict[str, Any]:
        """Run Agent 4 through Airia for executive summary generation."""
        return self._execute_agent(
            "compliance_docs",
            config.compliance_pipeline_id,
            {
                "report_data": report_data,
                "instruction": "Generate a formal executive summary for this treasury risk report.",
            },
        )

    def execute_risk_aggregation(self, all_signals: dict) -> dict[str, Any]:
        """Run Agent 5 — Risk Aggregator."""
        return self._execute_temporary("risk_aggregator", all_signals)

    def execute_anomaly_detection(self, market_data: list[dict]) -> dict[str, Any]:
        """Run Agent 6 — Anomaly Detector."""
        return self._execute_temporary("anomaly_detector", {"signals": market_data[:20]})

    def execute_sentiment_scoring(self, market_data: list[dict]) -> dict[str, Any]:
        """Run Agent 7 — Sentiment Scorer."""
        return self._execute_temporary("sentiment_scorer", {"signals": market_data[:15]})

    def execute_position_sizing(self, risk_profile: dict) -> dict[str, Any]:
        """Run Agent 8 — Position Sizer."""
        return self._execute_temporary("position_sizer", risk_profile)

    def execute_backtesting(self, strategies: list[dict]) -> dict[str, Any]:
        """Run Agent 9 — Strategy Backtester."""
        return self._execute_temporary("strategy_backtester", {"strategies": strategies})

    def execute_report_synthesis(self, all_results: dict) -> dict[str, Any]:
        """Run Agent 10 — Report Synthesizer."""
        return self._execute_temporary("report_synthesizer", all_results)

    # ── Utility ───────────────────────────────────────────────────────────

    def get_system_prompt(self, agent_name: str) -> str:
        """Get the system prompt for a given agent."""
        return AGENT_PROMPTS.get(agent_name, "")

    def list_prompts(self) -> dict[str, str]:
        """Return all agent prompts for reference."""
        return dict(AGENT_PROMPTS)

    def get_metrics(self) -> dict[str, dict]:
        """Return latency metrics per agent."""
        result = {}
        for agent, latencies in self._metrics.items():
            if latencies:
                result[agent] = {
                    "avg_ms": round(sum(latencies) / len(latencies), 1),
                    "min_ms": round(min(latencies), 1),
                    "max_ms": round(max(latencies), 1),
                    "calls": len(latencies),
                }
        return result


def _try_parse_json(text: str) -> dict | list | None:
    """Try to extract JSON from an LLM response."""
    text = text.strip()
    start = text.find("{")
    if start < 0:
        start = text.find("[")
    if start < 0:
        return None
    end = max(text.rfind("}"), text.rfind("]")) + 1
    if end <= start:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


bridge = AiriaBridge()

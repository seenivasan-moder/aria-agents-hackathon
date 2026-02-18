"""Airia SDK Bridge — pipeline execution with embedded system prompts.

Uses execute_temporary_assistant() when no pipeline IDs are configured,
injecting our system prompts directly. Falls back to execute_pipeline()
if pipeline IDs are set in .env.
"""

from __future__ import annotations

import json
import time
from typing import Any

from rich.console import Console

from src.config import config

console = Console()

# ═══════════════════════════════════════════════════════════════════════════════
# AIRIA PIPELINE SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

MARKET_INTELLIGENCE_PROMPT = """\
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

CORPORATE_CONTEXT_PROMPT = """\
Tu es l'Agent Sentinel-2 (Corporate Context Analyst).
Tu recois en entree le bilan de tresorerie et les factures impayees de l'entreprise.
Identifie les concentrations de risque par devise (ex: trop d'exposition au Dollar sans couverture).

SORTIE : Tu dois repondre EXCLUSIVEMENT en JSON :
{
  "net_exposure": {"USD": -500000, "EUR": 1200000},
  "risk_zones": ["High USD liability in Q3", "Unhedged Gold position"],
  "liquidity_status": "Optimal/Warning/Critical"
}"""

CONSENSUS_STRATEGY_PROMPT = """\
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
      "rationale": "Protection maximale contre la baisse de l'Euro detectee par Sentinel-1."
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

COMPLIANCE_DOCS_PROMPT = """\
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


# Map agent names to their prompts
AGENT_PROMPTS = {
    "market_intelligence": MARKET_INTELLIGENCE_PROMPT,
    "corporate_context": CORPORATE_CONTEXT_PROMPT,
    "consensus_strategy": CONSENSUS_STRATEGY_PROMPT,
    "compliance_docs": COMPLIANCE_DOCS_PROMPT,
}


# ═══════════════════════════════════════════════════════════════════════════════
# AIRIA BRIDGE
# ═══════════════════════════════════════════════════════════════════════════════

class AiriaBridge:
    """Wrapper around the Airia SDK with embedded pipeline prompts.

    Supports two modes:
    - Pipeline mode: uses pre-configured pipeline IDs (if set in .env)
    - Temporary assistant mode: uses execute_temporary_assistant() with
      embedded system prompts (no pipeline setup needed)
    """

    def __init__(self):
        self._client = None
        self._available: bool | None = None

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
            self._client = AiriaClient(api_key=config.airia_api_key)
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

    # ── Core execution ─────────────────────────────────────────────────────

    def _execute_temporary(
        self,
        agent_name: str,
        user_input: dict | str,
    ) -> dict[str, Any]:
        """Execute via temporary assistant with embedded system prompt."""
        client = self._get_client()
        if not client:
            return {"ok": False, "error": "Airia not available", "mode": "local-only"}

        prompt = AGENT_PROMPTS.get(agent_name, "")
        if not prompt:
            return {"ok": False, "error": f"No prompt for agent {agent_name}"}

        input_str = json.dumps(user_input) if isinstance(user_input, dict) else user_input

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

            console.print(f"  [dim]Airia {agent_name} (temp): {int(latency)}ms, {len(raw)} chars[/]")
            return {
                "ok": True,
                "result": raw,
                "parsed": parsed,
                "latency_ms": round(latency),
                "source": "airia-temp",
            }
        except Exception as e:
            console.print(f"  [yellow]Airia {agent_name} failed: {e}[/]")
            return {"ok": False, "error": str(e), "source": "airia"}

    def execute_pipeline(
        self,
        pipeline_id: str,
        user_input: dict | str,
        agent_name: str = "",
    ) -> dict[str, Any]:
        """Execute an Airia pipeline by ID. Returns parsed result or error."""
        client = self._get_client()
        if not client:
            return {"ok": False, "error": "Airia not available", "mode": "local-only"}
        if not pipeline_id:
            return {"ok": False, "error": f"Pipeline ID not configured for {agent_name}"}

        input_str = json.dumps(user_input) if isinstance(user_input, dict) else user_input

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

    # ── Utility ───────────────────────────────────────────────────────────

    def get_system_prompt(self, agent_name: str) -> str:
        """Get the system prompt for a given agent."""
        return AGENT_PROMPTS.get(agent_name, "")

    def list_prompts(self) -> dict[str, str]:
        """Return all 4 agent prompts for reference."""
        return dict(AGENT_PROMPTS)


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

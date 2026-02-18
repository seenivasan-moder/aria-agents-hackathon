"""Setup Airia infrastructure — creates the 4 Sentinel agent pipelines.

Run this once to provision all pipelines on the Airia platform.
Requires AIRIA_API_KEY in .env.

Usage:
    uv run python setup_airia_pipelines.py
"""

import json
import os
import uuid
import copy
from dotenv import load_dotenv

load_dotenv()

from airia import AiriaClient

KEY = os.getenv("AIRIA_API_KEY", "")
if not KEY or KEY == "your_airia_api_key_here":
    print("ERROR: Set AIRIA_API_KEY in .env first")
    exit(1)

client = AiriaClient(api_key=KEY)

# ── Agent definitions ──────────────────────────────────────────────────

AGENTS = [
    {
        "name": "Sentinel-1: Market Intelligence",
        "description": "Scans crypto, forex, and commodity markets. Evaluates risk scores, market regimes, and urgency.",
        "prompt": (
            "Tu es l'Agent Sentinel-1 (Market Intelligence).\n"
            "Ton role est d'analyser les donnees de marche fournies (Crypto, Forex, Commodities).\n"
            "Pour chaque actif, evalue :\n"
            "- Le score de risque (0-100) base sur la volatilite.\n"
            "- Le regime de marche (Trending, Range, Volatile).\n"
            "- L'urgence d'action.\n\n"
            "SORTIE : Tu dois repondre EXCLUSIVEMENT en JSON structure selon ce schema :\n"
            '{\n  "signals": [\n'
            '    {"asset": "BTC/USDT", "risk_score": 45, "regime": "Trending Up", "urgency": "Low"},\n'
            "    ...\n  ],\n"
            '  "global_sentiment": "Bullish/Bearish/Neutral"\n}'
        ),
        "env_key": "AIRIA_MARKET_PIPELINE_ID",
    },
    {
        "name": "Sentinel-2: Corporate Context",
        "description": "Analyzes corporate treasury data, identifies currency concentration risks, and evaluates liquidity status.",
        "prompt": (
            "Tu es l'Agent Sentinel-2 (Corporate Context Analyst).\n"
            "Tu recois en entree le bilan de tresorerie et les factures impayees de l'entreprise.\n"
            "Identifie les concentrations de risque par devise.\n\n"
            "SORTIE : Tu dois repondre EXCLUSIVEMENT en JSON :\n"
            '{\n  "net_exposure": {"USD": -500000, "EUR": 1200000},\n'
            '  "risk_zones": ["High USD liability in Q3", "Unhedged Gold position"],\n'
            '  "liquidity_status": "Optimal/Warning/Critical"\n}'
        ),
        "env_key": "AIRIA_CORPORATE_PIPELINE_ID",
    },
    {
        "name": "Sentinel-3: Consensus Strategy",
        "description": "Generates 3 hedging strategies (Conservative/Moderate/Aggressive) based on market signals and corporate exposure.",
        "prompt": (
            "Tu es l'Agent Sentinel-3 (Strategy Orchestrator).\n"
            "Inputs : Market Intelligence JSON + Corporate Context JSON\n\n"
            "TA MISSION : Generer 3 strategies de couverture (Hedging) :\n"
            "1. Conservatrice (Risque minime, cout eleve)\n"
            "2. Moderee (Equilibre)\n"
            "3. Agressive (Maximisation du profit, risque eleve)\n\n"
            "SORTIE : Repondre EXCLUSIVEMENT en JSON :\n"
            '{"strategies": [{"name": "Conservative", "description": "...", '
            '"instruments": ["FX Forward", "Commodity Futures"], '
            '"cost_estimate_pct": 0.45, "risk_reduction_pct": 85, '
            '"confidence": 89, "rationale": "..."}, '
            '{"name": "Moderate", ...}, {"name": "Aggressive", ...}]}'
        ),
        "env_key": "AIRIA_CONSENSUS_PIPELINE_ID",
    },
    {
        "name": "Sentinel-4: Compliance & Docs",
        "description": "Generates formal executive summaries for treasury risk reports. Professional Big Four tone.",
        "prompt": (
            "Tu es l'Agent Sentinel-4 (Compliance Officer).\n"
            "Tu dois rediger le resume executif du rapport de gestion des risques.\n"
            "Ton ton doit etre formel, precis et professionnel (style Big Four).\n\n"
            "TA MISSION :\n"
            "1. Resumer l'etat actuel des marches.\n"
            "2. Justifier la strategie recommandee par le consensus IA.\n"
            "3. Certifier que la piste d'audit est complete.\n\n"
            "SORTIE : Un texte structure avec des sections :\n"
            "[EXECUTIVE SUMMARY]\n[MARKET ANALYSIS]\n[INTERNAL EXPOSURE]\n"
            "[RECOMMENDATION]\n[AUDIT CERTIFICATION]"
        ),
        "env_key": "AIRIA_COMPLIANCE_PIPELINE_ID",
    },
]


def get_template():
    """Export an existing pipeline to use as template (camelCase format)."""
    pipelines = client.pipelines_config.get_pipelines_config()
    items = pipelines.items if hasattr(pipelines, "items") else []

    # Find a simple pipeline with AIOperation step
    for p in items:
        pid = p.id if hasattr(p, "id") else None
        if pid:
            try:
                defn = client.pipelines_config.export_pipeline_definition(str(pid))
                data = defn.model_dump(by_alias=True, exclude_none=True, mode="json")
                steps = data.get("agent", {}).get("steps", [])
                has_ai = any(s.get("stepType") == "AIOperation" for s in steps)
                if has_ai:
                    print(f"Using template: {p.name} ({pid})")
                    return data
            except Exception:
                continue

    raise RuntimeError("No suitable template pipeline found")


def build_definition(template: dict, agent_info: dict) -> dict:
    """Build a pipeline definition from template."""
    defn = copy.deepcopy(template)

    pipeline_id = str(uuid.uuid4())
    prompt_id = str(uuid.uuid4())
    input_step_id = str(uuid.uuid4())
    ai_step_id = str(uuid.uuid4())
    output_step_id = str(uuid.uuid4())
    input_handle = str(uuid.uuid4())
    ai_target_handle = str(uuid.uuid4())
    ai_source_handle = str(uuid.uuid4())
    output_handle = str(uuid.uuid4())

    model_id = template["models"][0]["id"]

    # Metadata
    defn["metadata"]["id"] = pipeline_id
    defn["metadata"]["tagline"] = agent_info["description"][:100]
    if "agentDescription" in defn["metadata"]:
        defn["metadata"]["agentDescription"] = agent_info["description"]
    defn["metadata"]["industry"] = "Finance"

    # Agent
    defn["agent"]["id"] = pipeline_id
    defn["agent"]["name"] = agent_info["name"]
    if "agentDescription" in defn["agent"]:
        defn["agent"]["agentDescription"] = agent_info["description"]
    defn["agent"]["tagline"] = agent_info["description"][:100]
    defn["agent"]["industry"] = "Finance"
    defn["agent"]["tags"] = ["sentinel", "treasury", "risk"]

    # Find template steps
    tmpl_steps = template["agent"]["steps"]
    tmpl_input = next((s for s in tmpl_steps if s["stepType"] == "inputStep"), tmpl_steps[0])
    tmpl_ai = next((s for s in tmpl_steps if s["stepType"] == "AIOperation"), tmpl_steps[0])
    tmpl_output = next((s for s in tmpl_steps if s["stepType"] == "outputStep"), tmpl_steps[-1])

    # Build steps
    input_step = copy.deepcopy(tmpl_input)
    input_step["id"] = input_step_id
    input_step["handles"] = [
        {"uuid": input_handle, "type": "source", "label": "", "tooltip": "", "x": 209.76, "y": 120.08}
    ]
    input_step["dependenciesObject"] = []

    ai_step = copy.deepcopy(tmpl_ai)
    ai_step["id"] = ai_step_id
    ai_step["handles"] = [
        {"uuid": ai_source_handle, "type": "source", "label": "", "tooltip": "", "x": 264.48, "y": 158.84},
        {"uuid": ai_target_handle, "type": "target", "label": "", "tooltip": "", "x": 264.48, "y": -9.12},
    ]
    ai_step["dependenciesObject"] = [
        {"parentId": input_step_id, "parentHandleId": input_handle, "handleId": ai_target_handle}
    ]
    ai_step["promptId"] = prompt_id
    ai_step["modelId"] = model_id
    ai_step["temperature"] = 0.3
    ai_step["toolIds"] = []
    ai_step["toolParamsJson"] = "{}"
    ai_step["includeDateTimeContext"] = True

    output_step = copy.deepcopy(tmpl_output)
    output_step["id"] = output_step_id
    output_step["handles"] = [
        {"uuid": output_handle, "type": "target", "label": "", "tooltip": "", "x": 173.28, "y": -9.12}
    ]
    output_step["dependenciesObject"] = [
        {"parentId": ai_step_id, "parentHandleId": ai_source_handle, "handleId": output_handle}
    ]

    defn["agent"]["steps"] = [input_step, ai_step, output_step]

    # Prompt
    defn["prompts"] = [
        {
            "name": prompt_id,
            "versionChangeDescription": f"System prompt for {agent_info['name']}",
            "promptMessage": agent_info["prompt"],
            "isAgentSpecific": True,
            "id": prompt_id,
        }
    ]

    defn["tools"] = []
    return defn


def main():
    print("Airia Sentinel — Pipeline Setup")
    print("=" * 50)

    template = get_template()
    created = []

    for agent_info in AGENTS:
        print(f"\nCreating: {agent_info['name']}...")
        definition = build_definition(template, agent_info)

        try:
            result = client.pipeline_import.create_agent_from_pipeline_definition(
                pipeline_definition=definition,
                agent_import_source="PlatformApi",
                conflict_resolution_strategy="RecreateExistingEntities",
                default_project_behavior="DefaultProject",
            )
            pid = str(result.pipeline_id) if hasattr(result, "pipeline_id") else "?"
            created.append({"name": agent_info["name"], "id": pid, "env_key": agent_info["env_key"]})
            print(f"  OK: {pid}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'=' * 50}")
    print("Add these to your .env file:\n")
    for item in created:
        print(f"{item['env_key']}={item['id']}")


if __name__ == "__main__":
    main()

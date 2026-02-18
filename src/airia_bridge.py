"""Airia SDK Bridge — wrapper for pipeline execution and swarm orchestration."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console

from src.config import config

console = Console()


class AiriaBridge:
    """Wrapper around the Airia SDK for pipeline execution."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not config.airia_api_key or config.airia_api_key == "your_airia_api_key_here":
                console.print("[yellow]Airia API key not configured — running in local-only mode[/]")
                return None
            try:
                from airia import AiriaClient
                self._client = AiriaClient(api_key=config.airia_api_key)
            except ImportError:
                console.print("[yellow]Airia SDK not installed — running in local-only mode[/]")
                return None
        return self._client

    def execute_pipeline(self, pipeline_id: str, user_input: dict | str) -> dict[str, Any]:
        """Execute an Airia pipeline and return the result."""
        client = self._get_client()
        if not client:
            return {"ok": False, "error": "Airia client not available", "mode": "local-only"}

        if not pipeline_id:
            return {"ok": False, "error": "Pipeline ID not configured"}

        input_str = json.dumps(user_input) if isinstance(user_input, dict) else user_input

        try:
            result = client.pipeline_execution.execute_pipeline(
                pipeline_id=pipeline_id,
                user_input=input_str,
            )
            return {"ok": True, "result": result.result if hasattr(result, "result") else str(result)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def execute_market_pipeline(self, pairs: list[str]) -> dict[str, Any]:
        return self.execute_pipeline(
            config.market_pipeline_id,
            {"pairs": pairs, "agent": "market_intelligence"},
        )

    def execute_corporate_pipeline(self, company_id: str) -> dict[str, Any]:
        return self.execute_pipeline(
            config.corporate_pipeline_id,
            {"company": company_id, "agent": "corporate_context"},
        )

    def execute_consensus_pipeline(self, market_data: str, corporate_data: str) -> dict[str, Any]:
        return self.execute_pipeline(
            config.consensus_pipeline_id,
            {"market_signals": market_data, "corporate_exposure": corporate_data, "agent": "consensus_strategy"},
        )

    def execute_compliance_pipeline(self, strategies: str) -> dict[str, Any]:
        return self.execute_pipeline(
            config.compliance_pipeline_id,
            {"strategies": strategies, "require_approval": True, "agent": "compliance_docs"},
        )

    @property
    def is_available(self) -> bool:
        return self._get_client() is not None


bridge = AiriaBridge()

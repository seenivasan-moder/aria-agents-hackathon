"""Sentinel configuration — Airia, LM Studio cluster, markets, thresholds."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ClusterNode:
    name: str
    url: str
    role: str
    default_model: str = ""


@dataclass
class SentinelConfig:
    # Airia SDK
    airia_api_key: str = field(default_factory=lambda: os.getenv("AIRIA_API_KEY", ""))
    market_pipeline_id: str = field(default_factory=lambda: os.getenv("AIRIA_MARKET_PIPELINE_ID", ""))
    corporate_pipeline_id: str = field(default_factory=lambda: os.getenv("AIRIA_CORPORATE_PIPELINE_ID", ""))
    consensus_pipeline_id: str = field(default_factory=lambda: os.getenv("AIRIA_CONSENSUS_PIPELINE_ID", ""))
    compliance_pipeline_id: str = field(default_factory=lambda: os.getenv("AIRIA_COMPLIANCE_PIPELINE_ID", ""))

    # LM Studio cluster (127.0.0.1, NEVER localhost)
    lm_nodes: list[ClusterNode] = field(default_factory=lambda: [
        ClusterNode(
            "M1", os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234"),
            "deep_analysis", default_model="qwen/qwen3-30b-a3b-2507",
        ),
    ])

    # Ollama
    ollama_nodes: list[ClusterNode] = field(default_factory=lambda: [
        ClusterNode(
            "OL1", os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
            "lightweight", default_model="qwen3:1.7b",
        ),
    ])

    # Market data
    mexc_api_key: str = field(default_factory=lambda: os.getenv("MEXC_API_KEY", ""))
    mexc_secret_key: str = field(default_factory=lambda: os.getenv("MEXC_SECRET_KEY", ""))
    watched_pairs: list[str] = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "EUR/USD", "GBP/USD",
        "JPY/USD", "XAU/USD", "XAG/USD", "CL/USD", "NG/USD",
    ])

    # Corporate simulation
    company_id: str = "ACME-CORP"
    base_currency: str = "EUR"
    mock_data_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "mock_corporate.json")

    # Thresholds
    risk_score_alert: float = 70.0
    volatility_alert: float = 2.5
    exposure_limit_pct: float = 25.0

    # Timeouts (seconds)
    connect_timeout: float = 5.0
    inference_timeout: float = 60.0
    health_timeout: float = 3.0

    # Inference params
    temperature: float = 0.3
    max_tokens: int = 4096

    # Database
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "sentinel.db")

    # Reports
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "reports")

    # HITL
    hitl_port: int = field(default_factory=lambda: int(os.getenv("HITL_PORT", "8900")))
    hitl_webhook_url: str = field(default_factory=lambda: os.getenv("HITL_WEBHOOK_URL", ""))

    def get_node(self, name: str) -> ClusterNode | None:
        for n in self.lm_nodes:
            if n.name == name:
                return n
        return None

    def get_ollama_node(self, name: str = "OL1") -> ClusterNode | None:
        for n in self.ollama_nodes:
            if n.name == name:
                return n
        return None


config = SentinelConfig()

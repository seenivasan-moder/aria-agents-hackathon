"""Pydantic models — Signals, Exposures, Strategies, Reports, Audit."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Regime(str, Enum):
    OVERSOLD = "oversold"
    NEUTRAL = "neutral"
    OVERBOUGHT = "overbought"
    CRISIS = "crisis"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# ── Agent 1: Market Intelligence ──────────────────────────────────────────


class MarketSignal(BaseModel):
    symbol: str
    asset_class: str = "crypto"  # crypto, forex, commodity
    price: float
    change_1h: float = 0.0
    change_24h: float = 0.0
    change_7d: float = 0.0
    volume_24h: float = 0.0
    volatility: float = 0.0
    funding_rate: float = 0.0
    risk_score: float = Field(default=0.0, ge=0, le=100)
    direction: Direction = Direction.NEUTRAL
    regime: Regime = Regime.NEUTRAL
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Agent 2: Corporate Context ────────────────────────────────────────────


class CurrencyPosition(BaseModel):
    currency: str
    cash_balance: float = 0.0
    receivables: float = 0.0
    payables: float = 0.0
    net_exposure: float = 0.0
    exposure_pct: float = 0.0


class CorporateExposure(BaseModel):
    company_id: str
    base_currency: str = "EUR"
    total_assets: float = 0.0
    total_liabilities: float = 0.0
    net_cash: float = 0.0
    positions: list[CurrencyPosition] = Field(default_factory=list)
    concentration_risks: list[str] = Field(default_factory=list)
    risk_score: float = Field(default=0.0, ge=0, le=100)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Agent 3: Consensus & Strategy ─────────────────────────────────────────


class HedgingStrategy(BaseModel):
    name: str  # e.g. "Conservative", "Moderate", "Aggressive"
    description: str
    instruments: list[str] = Field(default_factory=list)
    cost_estimate_pct: float = 0.0
    risk_reduction_pct: float = 0.0
    expected_return_pct: float = 0.0
    confidence: float = Field(default=0.0, ge=0, le=100)
    rationale: str = ""
    risk_level: RiskLevel = RiskLevel.MEDIUM


class ConsensusResult(BaseModel):
    strategies: list[HedgingStrategy] = Field(default_factory=list)
    recommended_index: int = 0  # index into strategies
    consensus_score: float = 0.0
    dissenting_views: list[str] = Field(default_factory=list)
    models_used: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Agent 4: Compliance & Docs ────────────────────────────────────────────


class AuditRecord(BaseModel):
    run_id: str
    step: str
    agent: str
    input_summary: str = ""
    output_summary: str = ""
    model_used: str = ""
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SentinelReport(BaseModel):
    run_id: str
    company_id: str
    market_signals: list[MarketSignal] = Field(default_factory=list)
    corporate_exposure: CorporateExposure | None = None
    consensus: ConsensusResult | None = None
    audit_trail: list[AuditRecord] = Field(default_factory=list)
    pdf_path: str = ""
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)

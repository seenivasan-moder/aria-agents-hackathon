"""Sentinel SQLite — Audit trail, signals, strategies, approvals."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from src.config import config


def get_connection() -> sqlite3.Connection:
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    conn = get_connection()

    tables = [
        """CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            asset_class TEXT DEFAULT 'crypto',
            price REAL,
            change_24h REAL,
            volatility REAL,
            risk_score REAL,
            direction TEXT,
            regime TEXT,
            raw_json TEXT,
            timestamp REAL DEFAULT 0
        )""",
        """CREATE TABLE IF NOT EXISTS exposures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            company_id TEXT NOT NULL,
            total_assets REAL,
            net_cash REAL,
            risk_score REAL,
            positions_json TEXT,
            concentration_risks TEXT,
            timestamp REAL DEFAULT 0
        )""",
        """CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            risk_level TEXT,
            confidence REAL,
            cost_estimate_pct REAL,
            risk_reduction_pct REAL,
            recommended INTEGER DEFAULT 0,
            rationale TEXT,
            timestamp REAL DEFAULT 0
        )""",
        """CREATE TABLE IF NOT EXISTS approvals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            strategy_name TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            approved_by TEXT DEFAULT '',
            comment TEXT DEFAULT '',
            timestamp REAL DEFAULT 0
        )""",
        """CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            step TEXT NOT NULL,
            agent TEXT NOT NULL,
            input_summary TEXT DEFAULT '',
            output_summary TEXT DEFAULT '',
            model_used TEXT DEFAULT '',
            latency_ms REAL DEFAULT 0,
            timestamp REAL DEFAULT 0
        )""",
    ]

    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_signals_run ON signals(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_exposures_run ON exposures(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_strategies_run ON strategies(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_audit_run ON audit_log(run_id)",
    ]

    for sql in tables:
        conn.execute(sql)
    for sql in indexes:
        conn.execute(sql)
    conn.commit()
    conn.close()


def save_signals(run_id: str, signals: list[dict]) -> int:
    conn = get_connection()
    count = 0
    for s in signals:
        conn.execute(
            """INSERT INTO signals (run_id, symbol, asset_class, price, change_24h,
               volatility, risk_score, direction, regime, raw_json, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, s.get("symbol"), s.get("asset_class", "crypto"),
             s.get("price", 0), s.get("change_24h", 0), s.get("volatility", 0),
             s.get("risk_score", 0), s.get("direction", "neutral"),
             s.get("regime", "neutral"), json.dumps(s), time.time()),
        )
        count += 1
    conn.commit()
    conn.close()
    return count


def save_exposure(run_id: str, exposure: dict) -> None:
    conn = get_connection()
    conn.execute(
        """INSERT INTO exposures (run_id, company_id, total_assets, net_cash,
           risk_score, positions_json, concentration_risks, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, exposure.get("company_id"), exposure.get("total_assets", 0),
         exposure.get("net_cash", 0), exposure.get("risk_score", 0),
         json.dumps(exposure.get("positions", [])),
         json.dumps(exposure.get("concentration_risks", [])), time.time()),
    )
    conn.commit()
    conn.close()


def save_strategies(run_id: str, strategies: list[dict], recommended_idx: int = 0) -> None:
    conn = get_connection()
    for i, s in enumerate(strategies):
        conn.execute(
            """INSERT INTO strategies (run_id, name, description, risk_level,
               confidence, cost_estimate_pct, risk_reduction_pct, recommended, rationale, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, s.get("name"), s.get("description"), s.get("risk_level", "medium"),
             s.get("confidence", 0), s.get("cost_estimate_pct", 0),
             s.get("risk_reduction_pct", 0), 1 if i == recommended_idx else 0,
             s.get("rationale", ""), time.time()),
        )
    conn.commit()
    conn.close()


def save_approval(run_id: str, strategy_name: str, status: str, approved_by: str = "", comment: str = "") -> None:
    conn = get_connection()
    conn.execute(
        """INSERT INTO approvals (run_id, strategy_name, status, approved_by, comment, timestamp)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (run_id, strategy_name, status, approved_by, comment, time.time()),
    )
    conn.commit()
    conn.close()


def save_audit(run_id: str, step: str, agent: str, input_summary: str = "",
               output_summary: str = "", model_used: str = "", latency_ms: float = 0) -> None:
    conn = get_connection()
    conn.execute(
        """INSERT INTO audit_log (run_id, step, agent, input_summary, output_summary,
           model_used, latency_ms, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, step, agent, input_summary, output_summary, model_used, latency_ms, time.time()),
    )
    conn.commit()
    conn.close()


def get_run_report(run_id: str) -> dict:
    conn = get_connection()
    signals = [dict(r) for r in conn.execute("SELECT * FROM signals WHERE run_id=?", (run_id,)).fetchall()]
    exposures = [dict(r) for r in conn.execute("SELECT * FROM exposures WHERE run_id=?", (run_id,)).fetchall()]
    strategies = [dict(r) for r in conn.execute("SELECT * FROM strategies WHERE run_id=?", (run_id,)).fetchall()]
    approvals = [dict(r) for r in conn.execute("SELECT * FROM approvals WHERE run_id=?", (run_id,)).fetchall()]
    audit = [dict(r) for r in conn.execute("SELECT * FROM audit_log WHERE run_id=? ORDER BY id", (run_id,)).fetchall()]
    conn.close()
    return {
        "run_id": run_id,
        "signals": signals,
        "exposures": exposures,
        "strategies": strategies,
        "approvals": approvals,
        "audit_log": audit,
    }


def get_pending_approvals() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM approvals WHERE status='pending' ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

"""Shared async HTTP client pool — adapted from JARVIS Turbo."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from src.config import config

_HTTP_POOL: httpx.AsyncClient | None = None
_POOL_LOCK = asyncio.Lock()
_METRICS: dict[str, list[float]] = {}


async def get_client() -> httpx.AsyncClient:
    global _HTTP_POOL
    if _HTTP_POOL is None or _HTTP_POOL.is_closed:
        async with _POOL_LOCK:
            if _HTTP_POOL is None or _HTTP_POOL.is_closed:
                _HTTP_POOL = httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=config.connect_timeout,
                        read=config.inference_timeout,
                        write=10.0,
                        pool=5.0,
                    ),
                    limits=httpx.Limits(
                        max_connections=20,
                        max_keepalive_connections=10,
                        keepalive_expiry=300,
                    ),
                )
    return _HTTP_POOL


async def close_pool() -> None:
    global _HTTP_POOL
    if _HTTP_POOL and not _HTTP_POOL.is_closed:
        await _HTTP_POOL.aclose()
        _HTTP_POOL = None


def track_latency(node: str, latency_ms: float) -> None:
    if node not in _METRICS:
        _METRICS[node] = []
    _METRICS[node].append(latency_ms)
    if len(_METRICS[node]) > 20:
        _METRICS[node] = _METRICS[node][-20:]


def get_avg_latency(node: str) -> float:
    vals = _METRICS.get(node, [])
    return sum(vals) / len(vals) if vals else 0.0


def get_all_metrics() -> dict[str, dict[str, float]]:
    result = {}
    for node, vals in _METRICS.items():
        if vals:
            result[node] = {
                "avg_ms": round(sum(vals) / len(vals), 1),
                "min_ms": round(min(vals), 1),
                "max_ms": round(max(vals), 1),
                "samples": len(vals),
            }
    return result

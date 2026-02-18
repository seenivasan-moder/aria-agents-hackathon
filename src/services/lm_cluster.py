"""LM Studio + Ollama cluster interface — query, consensus, health."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from src.config import config
from src.utils.http_pool import get_client, track_latency
from src.utils.retry import retry_request


async def query_lm(
    prompt: str,
    node_name: str = "M1",
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Query an LM Studio node."""
    node = config.get_node(node_name)
    if not node:
        return {"ok": False, "error": f"Unknown node: {node_name}"}

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        t0 = time.monotonic()
        r = await retry_request("POST", f"{node.url}/v1/chat/completions", json={
            "model": node.default_model,
            "messages": messages,
            "temperature": temperature or config.temperature,
            "max_tokens": max_tokens or config.max_tokens,
        })
        latency = (time.monotonic() - t0) * 1000
        track_latency(node_name, latency)
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return {"ok": True, "content": content, "model": node.default_model, "latency_ms": round(latency)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def query_ollama(
    prompt: str,
    node_name: str = "OL1",
    model: str | None = None,
    system: str | None = None,
) -> dict[str, Any]:
    """Query an Ollama node."""
    node = config.get_ollama_node(node_name)
    if not node:
        return {"ok": False, "error": f"Unknown Ollama node: {node_name}"}

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        t0 = time.monotonic()
        r = await retry_request("POST", f"{node.url}/api/chat", json={
            "model": model or node.default_model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": config.temperature, "num_predict": config.max_tokens},
        })
        latency = (time.monotonic() - t0) * 1000
        track_latency(node_name, latency)
        msg = r.json()["message"]
        content = msg.get("content", "") or msg.get("thinking", "")
        return {"ok": True, "content": content, "model": model or node.default_model, "latency_ms": round(latency)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def consensus(prompt: str, nodes: list[str] | None = None) -> dict[str, Any]:
    """Query multiple nodes in parallel and return all responses."""
    nodes = nodes or ["M1", "OL1"]
    responses: list[dict[str, Any]] = []

    async def _query(name: str) -> dict[str, Any]:
        if config.get_ollama_node(name):
            return await query_ollama(prompt, name)
        return await query_lm(prompt, name)

    results = await asyncio.gather(*[_query(n) for n in nodes], return_exceptions=True)

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            responses.append({"node": nodes[i], "ok": False, "error": str(r)})
        else:
            responses.append({"node": nodes[i], **r})

    return {
        "responses": responses,
        "nodes_queried": len(nodes),
        "nodes_ok": sum(1 for r in responses if r.get("ok")),
        "models_used": [r.get("model", "?") for r in responses if r.get("ok")],
    }


async def cluster_health() -> dict[str, Any]:
    """Check health of all cluster nodes."""
    client = await get_client()
    statuses = []

    for n in config.lm_nodes:
        try:
            t0 = time.monotonic()
            r = await client.get(f"{n.url}/v1/models", timeout=config.health_timeout)
            r.raise_for_status()
            latency = int((time.monotonic() - t0) * 1000)
            models = [m["id"] for m in r.json().get("data", [])]
            statuses.append({"node": n.name, "online": True, "latency_ms": latency, "models": models})
        except Exception:
            statuses.append({"node": n.name, "online": False})

    for n in config.ollama_nodes:
        try:
            t0 = time.monotonic()
            r = await client.get(f"{n.url}/api/tags", timeout=config.health_timeout)
            r.raise_for_status()
            latency = int((time.monotonic() - t0) * 1000)
            models = [m["name"] for m in r.json().get("models", [])]
            statuses.append({"node": n.name, "online": True, "latency_ms": latency, "models": models, "backend": "ollama"})
        except Exception:
            statuses.append({"node": n.name, "online": False, "backend": "ollama"})

    online = sum(1 for s in statuses if s.get("online"))
    return {"nodes": statuses, "online": online, "total": len(statuses)}

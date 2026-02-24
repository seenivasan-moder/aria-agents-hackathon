"""Dashboard WebSocket Server — extends HITL with real-time pipeline events.

Provides WebSocket endpoint for live dashboard updates and REST endpoints
for pipeline data. Extends the existing HITL FastAPI server.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from rich.console import Console

from src.config import config, PROJECT_ROOT
from src import database as db

console = Console()

# Create main app (includes HITL endpoints)
app = FastAPI(title="Airia Sentinel Dashboard", version="2.0.0")

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and mount HITL routes
from src.services.hitl_webhook import app as hitl_app
# Mount HITL routes directly on our app
for route in hitl_app.routes:
    app.routes.append(route)


# ── WebSocket Manager ─────────────────────────────────────────────────────

class ConnectionManager:
    """Manages WebSocket connections for live dashboard updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._pipeline_state: dict[str, Any] = {
            "run_id": None,
            "status": "idle",
            "agents": {},
            "signals": [],
            "consensus": None,
            "risk_profile": None,
            "cluster": {},
            "alerts": [],
            "audit_log": [],
            "started_at": None,
            "total_latency_ms": 0,
        }

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Send current state on connect
        await websocket.send_json({"type": "state", "data": self._pipeline_state})
        console.print(f"[cyan]Dashboard client connected[/] ({len(self.active_connections)} active)")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        console.print(f"[dim]Dashboard client disconnected[/] ({len(self.active_connections)} active)")

    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.active_connections.remove(conn)

    async def emit_agent_update(self, agent_name: str, status: str, data: dict | None = None):
        """Emit agent status update to all clients."""
        self._pipeline_state["agents"][agent_name] = {
            "status": status,
            "data": data,
            "updated_at": time.time(),
        }
        await self.broadcast({
            "type": "agent_update",
            "agent": agent_name,
            "status": status,
            "data": data,
        })

    async def emit_pipeline_event(self, event_type: str, data: dict):
        """Emit a pipeline lifecycle event."""
        if event_type == "pipeline_start":
            self._pipeline_state["run_id"] = data.get("run_id")
            self._pipeline_state["status"] = "running"
            self._pipeline_state["started_at"] = time.time()
            self._pipeline_state["agents"] = {}
        elif event_type == "pipeline_complete":
            self._pipeline_state["status"] = "complete"
            self._pipeline_state["total_latency_ms"] = data.get("total_latency_ms", 0)
        elif event_type == "signals_update":
            self._pipeline_state["signals"] = data.get("signals", [])
        elif event_type == "consensus_update":
            self._pipeline_state["consensus"] = data
        elif event_type == "risk_update":
            self._pipeline_state["risk_profile"] = data
        elif event_type == "alert":
            self._pipeline_state["alerts"].append(data)

        self._pipeline_state["audit_log"].append({
            "event": event_type,
            "timestamp": time.time(),
            "data_summary": str(data)[:200],
        })

        await self.broadcast({"type": event_type, "data": data})

    def get_state(self) -> dict:
        return self._pipeline_state


manager = ConnectionManager()


# ── WebSocket Endpoint ────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming commands from dashboard
            try:
                cmd = json.loads(data)
                if cmd.get("action") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
                elif cmd.get("action") == "get_state":
                    await websocket.send_json({"type": "state", "data": manager.get_state()})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ── REST Endpoints for Dashboard ──────────────────────────────────────────

@app.get("/dashboard")
async def serve_dashboard():
    """Serve the dashboard HTML file."""
    dashboard_path = PROJECT_ROOT / "dashboard" / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(str(dashboard_path), media_type="text/html")
    return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)


@app.get("/api/state")
async def get_pipeline_state():
    """Get current pipeline state."""
    return manager.get_state()


@app.get("/api/cluster")
async def get_cluster_status():
    """Get cluster health status."""
    try:
        from src.services.lm_cluster import cluster_health
        health = await cluster_health()
        return health
    except Exception as e:
        return {"error": str(e), "online": 0, "total": 0, "nodes": []}


@app.get("/api/signals")
async def get_signals():
    """Get latest market signals."""
    return {"signals": manager.get_state().get("signals", [])}


@app.get("/api/audit")
async def get_audit_log():
    """Get audit trail."""
    return {"log": manager.get_state().get("audit_log", [])}


@app.get("/api/stats")
async def get_stats():
    """Get dashboard statistics."""
    from src.airia_bridge import bridge
    state = manager.get_state()
    return {
        "total_agents": len(state.get("agents", {})),
        "pipeline_status": state.get("status", "idle"),
        "run_id": state.get("run_id"),
        "total_latency_ms": state.get("total_latency_ms", 0),
        "airia_available": bridge.is_available,
        "airia_stats": bridge.stats,
        "ws_clients": len(manager.active_connections),
    }


# ── Server Startup ────────────────────────────────────────────────────────

def start_dashboard_server():
    """Start the dashboard + HITL server with WebSocket support."""
    import uvicorn
    port = config.dashboard_port if hasattr(config, 'dashboard_port') else config.hitl_port
    console.print(f"[bold cyan]Starting Airia Sentinel Dashboard on port {port}...[/]")
    console.print(f"  Dashboard: http://127.0.0.1:{port}/dashboard")
    console.print(f"  WebSocket: ws://127.0.0.1:{port}/ws")
    console.print(f"  API: http://127.0.0.1:{port}/api/state")
    console.print(f"  HITL: http://127.0.0.1:{port}/pending")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

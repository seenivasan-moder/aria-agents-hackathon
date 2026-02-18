"""HITL Webhook — FastAPI server for human approval of strategies."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rich.console import Console

from src.config import config
from src import database as db

console = Console()

app = FastAPI(title="Airia Sentinel HITL", version="1.0.0")


class ApprovalRequest(BaseModel):
    run_id: str
    strategy_name: str
    approved_by: str = ""
    comment: str = ""


class RejectionRequest(BaseModel):
    run_id: str
    strategy_name: str
    rejected_by: str = ""
    reason: str = ""


@app.get("/")
async def root():
    return {"service": "Airia Sentinel HITL", "version": "1.0.0", "status": "running"}


@app.get("/pending")
async def get_pending():
    """List all pending approvals."""
    pending = db.get_pending_approvals()
    return {"pending": pending, "count": len(pending)}


@app.post("/approve")
async def approve_strategy(req: ApprovalRequest):
    """Approve a hedging strategy."""
    db.save_approval(req.run_id, req.strategy_name, "approved", req.approved_by, req.comment)
    console.print(f"[green]APPROVED[/] Strategy '{req.strategy_name}' for run {req.run_id} by {req.approved_by}")
    return {"status": "approved", "run_id": req.run_id, "strategy": req.strategy_name}


@app.post("/reject")
async def reject_strategy(req: RejectionRequest):
    """Reject a hedging strategy and escalate."""
    db.save_approval(req.run_id, req.strategy_name, "rejected", req.rejected_by, req.reason)
    console.print(f"[red]REJECTED[/] Strategy '{req.strategy_name}' for run {req.run_id} — reason: {req.reason}")
    return {"status": "rejected", "run_id": req.run_id, "strategy": req.strategy_name, "escalated": True}


@app.get("/report/{run_id}")
async def get_report(run_id: str):
    """Get full run report."""
    report = db.get_run_report(run_id)
    if not report.get("signals") and not report.get("strategies"):
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return report


def start_server():
    """Start the HITL webhook server."""
    import uvicorn
    console.print(f"[bold]Starting HITL server on port {config.hitl_port}...[/]")
    uvicorn.run(app, host="0.0.0.0", port=config.hitl_port, log_level="info")

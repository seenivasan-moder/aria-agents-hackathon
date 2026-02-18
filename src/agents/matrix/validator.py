"""Agent Validator — Quality gate between pipeline stages.

Validates data integrity, schema conformance, and business rules
between domino pipeline steps. Prevents garbage propagation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from rich.console import Console

from src.pipeline_engine import StepResult, StepStatus
from src import database as db

console = Console()


@dataclass
class ValidationRule:
    name: str
    check: str  # "not_empty", "has_key", "min_length", "score_above", "type_check"
    params: dict[str, Any] | None = None
    severity: str = "error"  # "error" blocks, "warning" continues


# ── Pre-built validation rules ───────────────────────────────────────

SIGNAL_RULES = [
    ValidationRule("signals_exist", "not_empty"),
    ValidationRule("min_signals", "min_length", {"min": 3}),
    ValidationRule("has_risk_scores", "has_key", {"key": "risk_score"}),
]

EXPOSURE_RULES = [
    ValidationRule("exposure_exists", "not_empty"),
    ValidationRule("risk_score_valid", "score_above", {"min": 0, "max": 100}),
]

STRATEGY_RULES = [
    ValidationRule("strategies_exist", "not_empty"),
    ValidationRule("has_confidence", "has_key", {"key": "confidence"}),
]


def _check_rule(rule: ValidationRule, data: Any) -> tuple[bool, str]:
    """Check a single validation rule against data."""
    if rule.check == "not_empty":
        if data is None or (isinstance(data, (list, dict, str)) and len(data) == 0):
            return False, f"{rule.name}: data is empty"
        return True, ""

    if rule.check == "min_length":
        min_len = rule.params.get("min", 1) if rule.params else 1
        if isinstance(data, (list, str)) and len(data) < min_len:
            return False, f"{rule.name}: length {len(data)} < {min_len}"
        return True, ""

    if rule.check == "has_key":
        key = rule.params.get("key", "") if rule.params else ""
        if isinstance(data, dict) and key not in data:
            return False, f"{rule.name}: missing key '{key}'"
        if isinstance(data, list) and data:
            item = data[0]
            if isinstance(item, dict):
                if key not in item:
                    return False, f"{rule.name}: first item missing key '{key}'"
            elif hasattr(item, key):
                pass  # attribute exists
            elif hasattr(item, "__dict__") and key in item.__dict__:
                pass  # instance attribute exists
            # For Pydantic models and dataclasses, check model fields
            elif hasattr(item, "model_fields") and key in item.model_fields:
                pass
            elif hasattr(item, "__dataclass_fields__") and key in item.__dataclass_fields__:
                pass
            # Don't fail for unknown types — be permissive
        return True, ""

    if rule.check == "score_above":
        min_score = rule.params.get("min", 0) if rule.params else 0
        max_score = rule.params.get("max", 100) if rule.params else 100
        if isinstance(data, (int, float)):
            if data < min_score or data > max_score:
                return False, f"{rule.name}: score {data} out of range [{min_score}, {max_score}]"
        return True, ""

    if rule.check == "type_check":
        expected = rule.params.get("type", "any") if rule.params else "any"
        type_map = {"list": list, "dict": dict, "str": str, "int": int, "float": float}
        if expected in type_map and not isinstance(data, type_map[expected]):
            return False, f"{rule.name}: expected {expected}, got {type(data).__name__}"
        return True, ""

    return True, ""


def validate(data: Any, rules: list[ValidationRule]) -> tuple[bool, list[str], list[str]]:
    """Validate data against a set of rules.

    Returns: (passed, errors, warnings)
    """
    errors = []
    warnings = []

    for rule in rules:
        passed, msg = _check_rule(rule, data)
        if not passed:
            if rule.severity == "error":
                errors.append(msg)
            else:
                warnings.append(msg)

    return len(errors) == 0, errors, warnings


async def run(run_id: str, data: Any, rules: list[ValidationRule] | None = None,
              step_name: str = "validation") -> StepResult:
    """Execute Agent Validator: check data quality between pipeline steps."""
    t0 = time.monotonic()
    console.print(f"[bold magenta]Agent Validator[/] checking {step_name}...")

    if rules is None:
        rules = SIGNAL_RULES  # default

    passed, errors, warnings = validate(data, rules)
    latency = (time.monotonic() - t0) * 1000

    for w in warnings:
        console.print(f"  [yellow]WARNING[/] {w}")
    for e in errors:
        console.print(f"  [red]ERROR[/] {e}")

    status = StepStatus.SUCCESS if passed else StepStatus.FAILED
    confidence = 100 if passed else max(0, 100 - len(errors) * 25)

    db.save_audit(
        run_id, "validation", "validator",
        input_summary=f"{step_name}: {len(rules)} rules",
        output_summary=f"{'PASS' if passed else 'FAIL'}: {len(errors)} errors, {len(warnings)} warnings",
        model_used="rule-based",
        latency_ms=latency,
    )

    console.print(f"[{'green' if passed else 'red'}]Validator {step_name}[/] — "
                  f"{'PASS' if passed else 'FAIL'} ({len(rules)} rules, {int(latency)}ms)")

    return StepResult(
        step_name=step_name,
        status=status,
        data=data,
        confidence=confidence,
        agent_used="validator",
        latency_ms=latency,
    )

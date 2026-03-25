"""Output parser: transform raw model completion to DetectionResult.

Spec: specifications/output-parser.md
"""

from __future__ import annotations

import json
import re
from typing import Any

from contextual_pii_tagger.entities import (
    DetectionResult,
    RiskLevel,
    SpanLabel,
)

_VALID_LABELS = {label.value for label in SpanLabel}
_VALID_RISKS = {level.value for level in RiskLevel}

_FALLBACK = DetectionResult(labels=frozenset(), risk=RiskLevel.LOW, rationale="")


def parse_output(raw_output: str) -> DetectionResult:
    """Parse raw model output into a validated DetectionResult.

    Total function: never raises, always returns a valid DetectionResult.
    """
    # Stage 1: JSON extraction
    data = _extract_json(raw_output)
    if data is None:
        return _FALLBACK

    # Stage 2-3: Field extraction with defaults
    raw_labels = data.get("labels", [])
    if not isinstance(raw_labels, list):
        raw_labels = []

    raw_risk = data.get("risk", "LOW")
    if raw_risk not in _VALID_RISKS:
        raw_risk = "LOW"

    raw_rationale = data.get("rationale", "")
    if not isinstance(raw_rationale, str):
        raw_rationale = ""

    # Stage 4: Label validation
    labels = _validate_labels(raw_labels)

    # Stage 5: Consistency enforcement
    risk = RiskLevel(raw_risk)
    rationale = raw_rationale

    if not labels:
        risk = RiskLevel.LOW
        rationale = ""
    elif risk == RiskLevel.LOW:
        rationale = ""
    elif risk in (RiskLevel.MEDIUM, RiskLevel.HIGH) and len(labels) >= 2 and not rationale:
        rationale = "Multiple quasi-identifiers detected."

    return DetectionResult(labels=labels, risk=risk, rationale=rationale)


def _extract_json(raw: str) -> dict[str, Any] | None:
    """Extract a JSON dict from raw output, with repair attempts."""
    if not raw or not raw.strip():
        return None

    # Try direct parse
    parsed = _try_parse(raw.strip())
    if isinstance(parsed, dict):
        return parsed

    # Try extracting from markdown code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if match:
        parsed = _try_parse(match.group(1).strip())
        if isinstance(parsed, dict):
            return parsed

    # Try extracting JSON object from surrounding text
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL)
    if match:
        parsed = _try_parse(match.group(0))
        if isinstance(parsed, dict):
            return parsed

    # Try repair
    repaired = _repair_json(raw.strip())
    if repaired is not None:
        parsed = _try_parse(repaired)
        if isinstance(parsed, dict):
            return parsed

    return None


def _try_parse(s: str) -> Any:
    """Try to parse a string as JSON."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


def _repair_json(s: str) -> str | None:
    """Attempt to fix common JSON malformations."""
    repaired = s

    # Replace single quotes with double quotes (simple heuristic)
    if "'" in repaired and '"' not in repaired:
        repaired = repaired.replace("'", '"')
    elif "'" in repaired:
        repaired = re.sub(r"'([^']*)'", r'"\1"', repaired)

    # Remove trailing commas before } or ]
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    # Try to close unclosed braces/brackets
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")
    if open_braces > 0:
        repaired += "}" * open_braces
    if open_brackets > 0:
        repaired += "]" * open_brackets

    if repaired != s:
        return repaired
    return None


def _validate_labels(raw_labels: list[Any]) -> frozenset[SpanLabel]:
    """Validate and deduplicate raw label strings into a SpanLabel frozenset."""
    valid: set[SpanLabel] = set()
    for raw in raw_labels:
        if isinstance(raw, str) and raw in _VALID_LABELS:
            valid.add(SpanLabel(raw))
    return frozenset(valid)

"""Core entities: SpanLabel, RiskLevel, DetectionResult.

Spec: specifications/entities.md §1-3
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class SpanLabel(StrEnum):
    """Tier 2 quasi-identifier categories."""

    LOCATION = "LOCATION"
    WORKPLACE = "WORKPLACE"
    ROUTINE = "ROUTINE"
    MEDICAL_CONTEXT = "MEDICAL-CONTEXT"
    DEMOGRAPHIC = "DEMOGRAPHIC"
    DEVICE_ID = "DEVICE-ID"
    CREDENTIAL = "CREDENTIAL"
    QUASI_ID = "QUASI-ID"


class RiskLevel(StrEnum):
    """Re-identification risk assessment."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class DetectionResult:
    """Output of a single detection invocation.

    Multilabel classification: reports which quasi-identifier categories
    are present, not where they appear in the text.
    """

    labels: frozenset[SpanLabel]
    risk: RiskLevel
    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.risk, RiskLevel):
            raise ValueError(f"DetectionResult.risk must be a RiskLevel, got {self.risk!r}")

        # Invariant: empty labels → LOW risk
        if len(self.labels) == 0 and self.risk != RiskLevel.LOW:
            raise ValueError("Empty labels requires risk LOW")

        # Invariant: LOW risk → empty rationale
        if self.risk == RiskLevel.LOW and self.rationale:
            raise ValueError("Risk LOW requires empty rationale")

        # Invariant: MEDIUM/HIGH with 2+ labels → non-empty rationale
        if (
            self.risk in (RiskLevel.MEDIUM, RiskLevel.HIGH)
            and len(self.labels) >= 2
            and not self.rationale
        ):
            raise ValueError(
                "Risk MEDIUM/HIGH with 2+ labels requires non-empty rationale"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": sorted(label.value for label in self.labels),
            "risk": self.risk.value,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DetectionResult:
        labels = frozenset(SpanLabel(v) for v in d.get("labels", []))
        return cls(
            labels=labels,
            risk=RiskLevel(d["risk"]),
            rationale=d.get("rationale", ""),
        )

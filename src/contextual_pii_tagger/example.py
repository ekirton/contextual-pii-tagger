"""Example and EvaluationReport entities.

Spec: specifications/entities.md §4-5
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from contextual_pii_tagger.entities import (
    RiskLevel,
    SpanLabel,
)

VALID_SPLITS = {"train", "validation", "test"}
VALID_DOMAINS = {"medical", "scheduling", "workplace", "personal"}
VALID_SOURCES = {"template", "llm-augmented", "hard-negative"}
_ID_PATTERN = re.compile(r"^(train|validation|test)-\d{5}$")


@dataclass(frozen=True)
class Example:
    """A single record in the dataset."""

    id: str
    text: str
    labels: frozenset[SpanLabel]
    risk: RiskLevel
    rationale: str
    is_hard_negative: bool
    split: str
    domain: str
    source: str

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Example.id must be non-empty")
        if not _ID_PATTERN.match(self.id):
            raise ValueError(
                f"Example.id must match {{split}}-{{5-digit number}}, got {self.id!r}"
            )
        if not self.text:
            raise ValueError("Example.text must be non-empty")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Example.split must be one of {VALID_SPLITS}, got {self.split!r}")
        if self.domain not in VALID_DOMAINS:
            raise ValueError(
                f"Example.domain must be one of {VALID_DOMAINS}, got {self.domain!r}"
            )
        if self.source not in VALID_SOURCES:
            raise ValueError(
                f"Example.source must be one of {VALID_SOURCES}, got {self.source!r}"
            )

        # Hard negative invariants
        if self.is_hard_negative:
            if self.labels:
                raise ValueError("Hard negative must have empty labels")
            if self.risk != RiskLevel.LOW:
                raise ValueError("Hard negative must have risk LOW")
            if self.rationale:
                raise ValueError("Hard negative must have empty rationale")
            if self.source != "hard-negative":
                raise ValueError("Hard negative must have source 'hard-negative'")

        # DetectionResult invariants
        if len(self.labels) == 0 and self.risk != RiskLevel.LOW:
            if not self.is_hard_negative:
                raise ValueError("Empty labels requires risk LOW")

        if self.risk == RiskLevel.LOW and self.rationale:
            raise ValueError("Risk LOW requires empty rationale")

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
            "id": self.id,
            "text": self.text,
            "labels": sorted(label.value for label in self.labels),
            "risk": self.risk.value,
            "rationale": self.rationale,
            "is_hard_negative": self.is_hard_negative,
            "split": self.split,
            "domain": self.domain,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Example:
        labels = frozenset(SpanLabel(v) for v in d.get("labels", []))
        return cls(
            id=d["id"],
            text=d["text"],
            labels=labels,
            risk=RiskLevel(d["risk"]),
            rationale=d.get("rationale", ""),
            is_hard_negative=d["is_hard_negative"],
            split=d["split"],
            domain=d["domain"],
            source=d["source"],
        )


@dataclass(frozen=True)
class EvaluationReport:
    """Output of the evaluation pipeline."""

    model_name: str
    test_set_size: int
    multilabel_f1: float
    f1_by_label: dict[SpanLabel, float]
    risk_accuracy: float
    false_negative_rate: float
    quasi_id_f1: float
    hard_negative_precision: float

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("EvaluationReport.model_name must be non-empty")
        if self.test_set_size <= 0:
            raise ValueError("EvaluationReport.test_set_size must be > 0")

        if len(self.f1_by_label) != len(SpanLabel):
            raise ValueError(
                f"f1_by_label must have {len(SpanLabel)} entries, "
                f"got {len(self.f1_by_label)}"
            )
        for label in SpanLabel:
            if label not in self.f1_by_label:
                raise ValueError(f"Missing SpanLabel {label.value} in f1_by_label")

        float_fields = {
            "multilabel_f1": self.multilabel_f1,
            "risk_accuracy": self.risk_accuracy,
            "false_negative_rate": self.false_negative_rate,
            "quasi_id_f1": self.quasi_id_f1,
            "hard_negative_precision": self.hard_negative_precision,
        }
        for name, val in float_fields.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0], got {val}")

        for label, val in self.f1_by_label.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"f1_by_label[{label.value}] must be in [0.0, 1.0], got {val}"
                )

        expected_avg = sum(self.f1_by_label.values()) / len(self.f1_by_label)
        if abs(self.multilabel_f1 - expected_avg) > 1e-6:
            raise ValueError(
                f"multilabel_f1 ({self.multilabel_f1}) must equal macro-average "
                f"of per-label values ({expected_avg})"
            )

"""RawExample: pre-split record for data generation stages.

Spec: specifications/data-generation.md §2.0
"""

from __future__ import annotations

from dataclasses import dataclass

from contextual_pii_tagger.entities import RiskLevel, SpanLabel
from contextual_pii_tagger.example import VALID_DOMAINS, VALID_SOURCES


@dataclass(frozen=True)
class RawExample:
    """A pre-split record produced before the orchestrator assigns id and split."""

    text: str
    labels: frozenset[SpanLabel]
    risk: RiskLevel
    rationale: str
    is_hard_negative: bool
    domain: str
    source: str

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("RawExample.text must be non-empty")
        if self.domain not in VALID_DOMAINS:
            raise ValueError(
                f"RawExample.domain must be one of {VALID_DOMAINS}, got {self.domain!r}"
            )
        if self.source not in VALID_SOURCES:
            raise ValueError(
                f"RawExample.source must be one of {VALID_SOURCES}, got {self.source!r}"
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
            and not self.rationale
        ):
            raise ValueError(
                "Risk MEDIUM/HIGH requires non-empty rationale"
            )

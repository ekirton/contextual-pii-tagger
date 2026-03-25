"""Human review workflow (F-06, Stage 5).

Spec: specifications/data-generation.md §6
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from contextual_pii_tagger.entities import RiskLevel, SpanLabel
from contextual_pii_tagger.example import Example


@dataclass(frozen=True)
class Correction:
    """A single correction to apply to an Example by ID.

    Only non-None fields are applied. This allows partial corrections
    (e.g., fix only the risk level without changing labels).
    """

    id: str
    labels: Optional[frozenset[SpanLabel]] = None
    risk: Optional[RiskLevel] = None
    rationale: Optional[str] = None
    is_hard_negative: Optional[bool] = None
    source: Optional[str] = None


def select_review_sample(
    dataset: list[Example],
    ratio: float,
    seed: int | None = None,
) -> list[Example]:
    """Select a random sample proportionally from each split.

    Spec: specifications/data-generation.md §6.1

    REQUIRES:
    - ratio is in (0.0, 1.0).
    - dataset contains Examples with valid split fields.

    ENSURES:
    - Returns ratio * len(dataset) examples (rounded per split).
    - Sample is drawn proportionally from each split.
    - Selection is random (not stratified by domain or label).
    """
    rng = random.Random(seed)

    by_split: dict[str, list[Example]] = {}
    for ex in dataset:
        by_split.setdefault(ex.split, []).append(ex)

    sample: list[Example] = []
    for split, examples in sorted(by_split.items()):
        n = max(1, round(len(examples) * ratio))
        sample.extend(rng.sample(examples, min(n, len(examples))))

    return sample


def apply_corrections(
    dataset: list[Example],
    corrections: list[Correction],
) -> list[Example]:
    """Apply corrections to matching Examples by ID.

    Spec: specifications/data-generation.md §6.2

    REQUIRES:
    - Each Correction references an Example by id.

    ENSURES:
    - Corrected fields are applied to matching Example records.
    - No Examples are added or removed.
    - Split assignments are not changed.
    - All corrected Examples satisfy Example invariants.

    RAISES:
    - KeyError if a Correction targets a non-existent ID.
    - ValueError if a corrected Example violates invariants.
    """
    if not corrections:
        return list(dataset)

    index: dict[str, int] = {ex.id: i for i, ex in enumerate(dataset)}

    for correction in corrections:
        if correction.id not in index:
            raise KeyError(
                f"Correction targets non-existent id: {correction.id!r}"
            )

    result = list(dataset)

    for correction in corrections:
        i = index[correction.id]
        original = result[i]

        fields: dict[str, object] = {
            "id": original.id,
            "text": original.text,
            "labels": original.labels,
            "risk": original.risk,
            "rationale": original.rationale,
            "is_hard_negative": original.is_hard_negative,
            "split": original.split,
            "domain": original.domain,
            "source": original.source,
        }

        if correction.labels is not None:
            fields["labels"] = correction.labels
        if correction.risk is not None:
            fields["risk"] = correction.risk
        if correction.rationale is not None:
            fields["rationale"] = correction.rationale
        if correction.is_hard_negative is not None:
            fields["is_hard_negative"] = correction.is_hard_negative
        if correction.source is not None:
            fields["source"] = correction.source

        # Example.__init__ enforces invariants
        result[i] = Example(**fields)

    return result

"""Tests for human review workflow (F-06).

All expectations derived from specifications/data-generation.md §6.
"""

from __future__ import annotations

import pytest

from contextual_pii_tagger.entities import RiskLevel, SpanLabel
from contextual_pii_tagger.example import Example
from contextual_pii_tagger.data.human_review import (
    Correction,
    apply_corrections,
    select_review_sample,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_example(
    idx: int,
    split: str = "train",
    label: SpanLabel = SpanLabel.LOCATION,
) -> Example:
    return Example(
        id=f"{split}-{idx:05d}",
        text=f"Example text {idx} for {split}",
        labels=frozenset({label}),
        risk=RiskLevel.MEDIUM,
        rationale="",
        is_hard_negative=False,
        split=split,
        domain="medical",
        source="template",
    )


def _make_hard_negative(idx: int, split: str = "train") -> Example:
    return Example(
        id=f"{split}-{idx:05d}",
        text=f"Historical reference {idx}",
        labels=frozenset(),
        risk=RiskLevel.LOW,
        rationale="",
        is_hard_negative=True,
        split=split,
        domain="medical",
        source="hard-negative",
    )


def _make_dataset(
    train: int = 80,
    validation: int = 10,
    test: int = 10,
) -> list[Example]:
    examples: list[Example] = []
    for i in range(1, train + 1):
        examples.append(_make_example(i, "train"))
    for i in range(1, validation + 1):
        examples.append(_make_example(i, "validation"))
    for i in range(1, test + 1):
        examples.append(_make_example(i, "test"))
    return examples


# ── §6.1: select_review_sample ──────────────────────────────────────────


class TestSelectReviewSample:
    """ENSURES: random 1% sample, drawn proportionally from each split."""

    def test_returns_correct_count(self):
        """1% of 1000 examples = 10."""
        dataset = _make_dataset(train=800, validation=100, test=100)
        sample = select_review_sample(dataset, ratio=0.01)
        assert len(sample) == 10

    def test_proportional_across_splits(self):
        """Sample is drawn proportionally from each split."""
        dataset = _make_dataset(train=800, validation=100, test=100)
        sample = select_review_sample(dataset, ratio=0.01)
        splits = {"train": 0, "validation": 0, "test": 0}
        for ex in sample:
            splits[ex.split] += 1
        assert splits["train"] == 8
        assert splits["validation"] == 1
        assert splits["test"] == 1

    def test_returns_example_objects(self):
        """Sample contains valid Example instances."""
        dataset = _make_dataset()
        sample = select_review_sample(dataset, ratio=0.01)
        for ex in sample:
            assert isinstance(ex, Example)

    def test_sample_is_subset_of_dataset(self):
        """Every sampled example exists in the original dataset."""
        dataset = _make_dataset(train=800, validation=100, test=100)
        sample = select_review_sample(dataset, ratio=0.01)
        dataset_ids = {ex.id for ex in dataset}
        for ex in sample:
            assert ex.id in dataset_ids

    def test_random_selection_varies_with_seed(self):
        """Different seeds produce different samples."""
        dataset = _make_dataset(train=800, validation=100, test=100)
        s1 = select_review_sample(dataset, ratio=0.01, seed=42)
        s2 = select_review_sample(dataset, ratio=0.01, seed=99)
        ids1 = {ex.id for ex in s1}
        ids2 = {ex.id for ex in s2}
        assert ids1 != ids2

    def test_deterministic_with_same_seed(self):
        """Same seed produces same sample."""
        dataset = _make_dataset(train=800, validation=100, test=100)
        s1 = select_review_sample(dataset, ratio=0.01, seed=42)
        s2 = select_review_sample(dataset, ratio=0.01, seed=42)
        ids1 = [ex.id for ex in s1]
        ids2 = [ex.id for ex in s2]
        assert ids1 == ids2

    def test_small_split_rounds_to_at_least_one(self):
        """Even a small split gets at least one sample if ratio allows."""
        dataset = _make_dataset(train=80, validation=10, test=10)
        sample = select_review_sample(dataset, ratio=0.10)
        splits = {ex.split for ex in sample}
        assert "validation" in splits
        assert "test" in splits


# ── §6.2: apply_corrections ─────────────────────────────────────────────


class TestApplyCorrections:
    """ENSURES: corrections applied by ID; no adds/removes; invariants hold."""

    def test_applies_label_correction(self):
        """A correction can change labels on a matching example."""
        dataset = _make_dataset(train=10, validation=0, test=0)
        target = dataset[0]
        corrections = [
            Correction(
                id=target.id,
                labels=frozenset({SpanLabel.WORKPLACE, SpanLabel.LOCATION}),
                risk=RiskLevel.HIGH,
                rationale="Workplace and location identify the person.",
            ),
        ]
        result = apply_corrections(dataset, corrections)
        corrected = next(ex for ex in result if ex.id == target.id)
        assert corrected.labels == frozenset({SpanLabel.WORKPLACE, SpanLabel.LOCATION})
        assert corrected.risk == RiskLevel.HIGH
        assert corrected.rationale == "Workplace and location identify the person."

    def test_no_examples_added_or_removed(self):
        """Dataset size is unchanged after corrections."""
        dataset = _make_dataset(train=10, validation=5, test=5)
        corrections = [
            Correction(
                id=dataset[0].id,
                labels=frozenset({SpanLabel.ROUTINE}),
                risk=RiskLevel.MEDIUM,
                rationale="",
            ),
        ]
        result = apply_corrections(dataset, corrections)
        assert len(result) == len(dataset)

    def test_split_assignments_unchanged(self):
        """Split assignments are not modified by corrections."""
        dataset = _make_dataset(train=10, validation=5, test=5)
        corrections = [
            Correction(
                id=dataset[0].id,
                labels=frozenset({SpanLabel.DEMOGRAPHIC}),
                risk=RiskLevel.MEDIUM,
                rationale="",
            ),
        ]
        result = apply_corrections(dataset, corrections)
        for orig, corrected in zip(dataset, result):
            assert orig.split == corrected.split

    def test_uncorrected_examples_unchanged(self):
        """Examples not targeted by corrections remain identical."""
        dataset = _make_dataset(train=10, validation=0, test=0)
        corrections = [
            Correction(
                id=dataset[0].id,
                labels=frozenset({SpanLabel.ROUTINE}),
                risk=RiskLevel.MEDIUM,
                rationale="",
            ),
        ]
        result = apply_corrections(dataset, corrections)
        for orig, corrected in zip(dataset[1:], result[1:]):
            assert orig == corrected

    def test_corrected_example_satisfies_invariants(self):
        """Corrections must produce valid Examples (invariants enforced)."""
        dataset = _make_dataset(train=10, validation=0, test=0)
        corrections = [
            Correction(
                id=dataset[0].id,
                labels=frozenset({SpanLabel.LOCATION, SpanLabel.WORKPLACE}),
                risk=RiskLevel.HIGH,
                rationale="Combined quasi-identifiers.",
            ),
        ]
        result = apply_corrections(dataset, corrections)
        corrected = next(ex for ex in result if ex.id == dataset[0].id)
        assert isinstance(corrected, Example)

    def test_invalid_correction_raises(self):
        """Correction that violates invariants raises ValueError."""
        dataset = _make_dataset(train=10, validation=0, test=0)
        corrections = [
            Correction(
                id=dataset[0].id,
                labels=frozenset(),
                risk=RiskLevel.HIGH,  # empty labels + HIGH = invariant violation
                rationale="Should fail.",
            ),
        ]
        with pytest.raises(ValueError):
            apply_corrections(dataset, corrections)

    def test_unknown_id_raises(self):
        """Correction targeting a non-existent ID raises KeyError."""
        dataset = _make_dataset(train=10, validation=0, test=0)
        corrections = [
            Correction(
                id="train-99999",
                labels=frozenset({SpanLabel.LOCATION}),
                risk=RiskLevel.MEDIUM,
                rationale="",
            ),
        ]
        with pytest.raises(KeyError):
            apply_corrections(dataset, corrections)

    def test_empty_corrections_returns_copy(self):
        """No corrections returns the dataset unchanged."""
        dataset = _make_dataset(train=10, validation=0, test=0)
        result = apply_corrections(dataset, [])
        assert len(result) == len(dataset)
        for orig, res in zip(dataset, result):
            assert orig == res

    def test_hard_negative_reclassification(self):
        """A hard negative can be corrected to a regular example."""
        hn = _make_hard_negative(idx=11, split="train")
        dataset = _make_dataset(train=10, validation=0, test=0) + [hn]
        corrections = [
            Correction(
                id=hn.id,
                labels=frozenset({SpanLabel.LOCATION}),
                risk=RiskLevel.MEDIUM,
                rationale="",
                is_hard_negative=False,
                source="template",
            ),
        ]
        result = apply_corrections(dataset, corrections)
        corrected = next(ex for ex in result if ex.id == hn.id)
        assert not corrected.is_hard_negative
        assert SpanLabel.LOCATION in corrected.labels


# ── Correction dataclass ─────────────────────────────────────────────────


class TestCorrection:
    """ENSURES: Correction has required fields."""

    def test_requires_id(self):
        """id is required."""
        with pytest.raises(TypeError):
            Correction(
                labels=frozenset({SpanLabel.LOCATION}),
                risk=RiskLevel.MEDIUM,
                rationale="",
            )

    def test_optional_fields_default_none(self):
        """Fields besides id default to None (no change)."""
        c = Correction(id="train-00001")
        assert c.labels is None
        assert c.risk is None
        assert c.rationale is None
        assert c.is_hard_negative is None
        assert c.source is None

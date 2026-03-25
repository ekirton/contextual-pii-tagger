"""Tests for Example and EvaluationReport entities.

All expectations derived from specifications/entities.md §4-5.
"""

import pytest

from contextual_pii_tagger.entities import (
    RiskLevel,
    SpanLabel,
)
from contextual_pii_tagger.example import EvaluationReport, Example


# ── Example (entities.md §4) ──────────────────────────────────────────────


class TestExample:
    """REQUIRES/ENSURES on construction; hard negative invariant."""

    def test_valid_construction(self):
        ex = Example(
            id="train-00001",
            text="I work at Providence Hospital",
            labels=frozenset({SpanLabel.WORKPLACE}),
            risk=RiskLevel.MEDIUM,
            rationale="",
            is_hard_negative=False,
            split="train",
            domain="workplace",
            source="template",
        )
        assert ex.id == "train-00001"
        assert ex.split == "train"

    def test_empty_id_raises(self):
        """REQUIRES: id non-empty."""
        with pytest.raises(ValueError):
            Example(
                id="",
                text="some text",
                labels=frozenset(),
                risk=RiskLevel.LOW,
                rationale="",
                is_hard_negative=True,
                split="train",
                domain="medical",
                source="hard-negative",
            )

    def test_empty_text_raises(self):
        """REQUIRES: text non-empty."""
        with pytest.raises(ValueError):
            Example(
                id="train-00001",
                text="",
                labels=frozenset(),
                risk=RiskLevel.LOW,
                rationale="",
                is_hard_negative=True,
                split="train",
                domain="medical",
                source="hard-negative",
            )

    def test_invalid_split_raises(self):
        """REQUIRES: split in {train, validation, test}."""
        with pytest.raises(ValueError):
            Example(
                id="train-00001",
                text="some text",
                labels=frozenset(),
                risk=RiskLevel.LOW,
                rationale="",
                is_hard_negative=True,
                split="holdout",
                domain="medical",
                source="hard-negative",
            )

    def test_invalid_domain_raises(self):
        """REQUIRES: domain in {medical, scheduling, workplace, personal}."""
        with pytest.raises(ValueError):
            Example(
                id="train-00001",
                text="some text",
                labels=frozenset(),
                risk=RiskLevel.LOW,
                rationale="",
                is_hard_negative=True,
                split="train",
                domain="legal",
                source="hard-negative",
            )

    def test_invalid_source_raises(self):
        """REQUIRES: source in {template, llm-augmented, hard-negative}."""
        with pytest.raises(ValueError):
            Example(
                id="train-00001",
                text="some text",
                labels=frozenset(),
                risk=RiskLevel.LOW,
                rationale="",
                is_hard_negative=True,
                split="train",
                domain="medical",
                source="manual",
            )

    def test_hard_negative_must_have_empty_labels(self):
        """ENSURES: hard negative → labels empty."""
        with pytest.raises(ValueError):
            Example(
                id="train-00001",
                text="I work at Providence Hospital",
                labels=frozenset({SpanLabel.WORKPLACE}),
                risk=RiskLevel.LOW,
                rationale="",
                is_hard_negative=True,
                split="train",
                domain="workplace",
                source="hard-negative",
            )

    def test_hard_negative_must_have_low_risk(self):
        """ENSURES: hard negative → risk LOW."""
        with pytest.raises(ValueError):
            Example(
                id="train-00001",
                text="some text",
                labels=frozenset(),
                risk=RiskLevel.MEDIUM,
                rationale="",
                is_hard_negative=True,
                split="train",
                domain="medical",
                source="hard-negative",
            )

    def test_hard_negative_must_have_empty_rationale(self):
        """ENSURES: hard negative → rationale empty."""
        with pytest.raises(ValueError):
            Example(
                id="train-00001",
                text="some text",
                labels=frozenset(),
                risk=RiskLevel.LOW,
                rationale="should be empty",
                is_hard_negative=True,
                split="train",
                domain="medical",
                source="hard-negative",
            )

    def test_hard_negative_must_have_hard_negative_source(self):
        """ENSURES: hard negative → source hard-negative."""
        with pytest.raises(ValueError):
            Example(
                id="train-00001",
                text="some text",
                labels=frozenset(),
                risk=RiskLevel.LOW,
                rationale="",
                is_hard_negative=True,
                split="train",
                domain="medical",
                source="template",
            )

    def test_valid_hard_negative(self):
        ex = Example(
            id="train-00001",
            text="The Battle of Gettysburg took place in July 1863",
            labels=frozenset(),
            risk=RiskLevel.LOW,
            rationale="",
            is_hard_negative=True,
            split="train",
            domain="personal",
            source="hard-negative",
        )
        assert ex.is_hard_negative is True

    def test_id_pattern(self):
        """MAINTAINS: id follows {split}-{zero-padded-number}."""
        with pytest.raises(ValueError):
            Example(
                id="bad_id_format",
                text="some text",
                labels=frozenset(),
                risk=RiskLevel.LOW,
                rationale="",
                is_hard_negative=True,
                split="train",
                domain="medical",
                source="hard-negative",
            )

    def test_serialization_round_trip(self):
        ex = Example(
            id="test-00042",
            text="I work at Providence Hospital",
            labels=frozenset({SpanLabel.WORKPLACE}),
            risk=RiskLevel.MEDIUM,
            rationale="",
            is_hard_negative=False,
            split="test",
            domain="workplace",
            source="template",
        )
        d = ex.to_dict()
        restored = Example.from_dict(d)
        assert restored.id == ex.id
        assert restored.text == ex.text
        assert restored.split == ex.split
        assert restored.is_hard_negative == ex.is_hard_negative
        assert restored.labels == ex.labels


# ── EvaluationReport (entities.md §5) ─────────────────────────────────────


class TestEvaluationReport:
    """REQUIRES/ENSURES on construction."""

    def _make_by_label(self, value: float) -> dict:
        return {label: value for label in SpanLabel}

    def test_valid_construction(self):
        by_label = self._make_by_label(0.80)
        report = EvaluationReport(
            model_name="test-model",
            test_set_size=5000,
            multilabel_f1=0.80,
            f1_by_label=by_label,
            risk_accuracy=0.85,
            false_negative_rate=0.05,
            quasi_id_f1=0.70,
            hard_negative_precision=0.95,
        )
        assert report.model_name == "test-model"
        assert report.test_set_size == 5000

    def test_empty_model_name_raises(self):
        """REQUIRES: model_name non-empty."""
        by_label = self._make_by_label(0.80)
        with pytest.raises(ValueError):
            EvaluationReport(
                model_name="",
                test_set_size=5000,
                multilabel_f1=0.80,
                f1_by_label=by_label,
                risk_accuracy=0.85,
                false_negative_rate=0.05,
                quasi_id_f1=0.70,
                hard_negative_precision=0.95,
            )

    def test_zero_test_set_size_raises(self):
        """REQUIRES: test_set_size > 0."""
        by_label = self._make_by_label(0.80)
        with pytest.raises(ValueError):
            EvaluationReport(
                model_name="test",
                test_set_size=0,
                multilabel_f1=0.80,
                f1_by_label=by_label,
                risk_accuracy=0.85,
                false_negative_rate=0.05,
                quasi_id_f1=0.70,
                hard_negative_precision=0.95,
            )

    def test_by_label_must_have_eight_entries(self):
        """ENSURES: f1_by_label has exactly 8 entries."""
        partial = {SpanLabel.LOCATION: 0.80, SpanLabel.WORKPLACE: 0.80}
        with pytest.raises(ValueError):
            EvaluationReport(
                model_name="test",
                test_set_size=5000,
                multilabel_f1=0.80,
                f1_by_label=partial,
                risk_accuracy=0.85,
                false_negative_rate=0.05,
                quasi_id_f1=0.70,
                hard_negative_precision=0.95,
            )

    def test_multilabel_f1_must_equal_macro_average(self):
        """ENSURES: multilabel_f1 == macro-average of per-label values."""
        by_label = {label: float(i) / 10 for i, label in enumerate(SpanLabel)}
        wrong_avg = 0.99
        with pytest.raises(ValueError):
            EvaluationReport(
                model_name="test",
                test_set_size=5000,
                multilabel_f1=wrong_avg,
                f1_by_label=by_label,
                risk_accuracy=0.85,
                false_negative_rate=0.05,
                quasi_id_f1=0.70,
                hard_negative_precision=0.95,
            )

    def test_float_out_of_range_raises(self):
        """ENSURES: all floats in [0.0, 1.0]."""
        by_label = self._make_by_label(0.80)
        with pytest.raises(ValueError):
            EvaluationReport(
                model_name="test",
                test_set_size=5000,
                multilabel_f1=0.80,
                f1_by_label=by_label,
                risk_accuracy=1.5,  # out of range
                false_negative_rate=0.05,
                quasi_id_f1=0.70,
                hard_negative_precision=0.95,
            )

    def test_immutable(self):
        """MAINTAINS: immutable after construction."""
        by_label = self._make_by_label(0.80)
        report = EvaluationReport(
            model_name="test",
            test_set_size=5000,
            multilabel_f1=0.80,
            f1_by_label=by_label,
            risk_accuracy=0.85,
            false_negative_rate=0.05,
            quasi_id_f1=0.70,
            hard_negative_precision=0.95,
        )
        with pytest.raises(AttributeError):
            report.model_name = "changed"

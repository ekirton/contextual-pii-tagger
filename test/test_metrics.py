"""Tests for evaluation metrics.

All expectations derived from specifications/evaluation.md §2-6.
"""

import pytest

from contextual_pii_tagger.entities import (
    DetectionResult,
    RiskLevel,
    SpanLabel,
)
from contextual_pii_tagger.eval.metrics import (
    compute_false_negative_rate,
    compute_hard_negative_precision,
    compute_multilabel_f1,
    compute_quasi_id_f1,
    compute_risk_accuracy,
)
from contextual_pii_tagger.example import Example


# ── §3: Risk Score Accuracy ───────────────────────────────────────────────


class TestRiskAccuracy:
    """ENSURES: fraction of exact matches."""

    def test_all_correct(self):
        preds = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
        gts = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
        assert compute_risk_accuracy(preds, gts) == 1.0

    def test_all_wrong(self):
        preds = [RiskLevel.LOW, RiskLevel.LOW, RiskLevel.LOW]
        gts = [RiskLevel.HIGH, RiskLevel.HIGH, RiskLevel.HIGH]
        assert compute_risk_accuracy(preds, gts) == 0.0

    def test_partial(self):
        preds = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.LOW]
        gts = [RiskLevel.LOW, RiskLevel.HIGH, RiskLevel.HIGH, RiskLevel.LOW]
        assert compute_risk_accuracy(preds, gts) == 0.75

    def test_range(self):
        """ENSURES: range [0.0, 1.0]."""
        preds = [RiskLevel.LOW]
        gts = [RiskLevel.LOW]
        result = compute_risk_accuracy(preds, gts)
        assert 0.0 <= result <= 1.0


# ── §4: False Negative Rate ───────────────────────────────────────────────


class TestFalseNegativeRate:
    """ENSURES: missed / pii_examples."""

    def test_no_false_negatives(self):
        """All PII texts correctly identified as having PII."""
        preds = [frozenset({SpanLabel.LOCATION})]
        gts = [frozenset({SpanLabel.LOCATION})]
        assert compute_false_negative_rate(preds, gts) == 0.0

    def test_all_false_negatives(self):
        """All PII texts classified as clean."""
        preds = [frozenset(), frozenset()]
        gts = [frozenset({SpanLabel.LOCATION}), frozenset({SpanLabel.WORKPLACE})]
        assert compute_false_negative_rate(preds, gts) == 1.0

    def test_partial_false_negatives(self):
        preds = [frozenset({SpanLabel.LOCATION}), frozenset()]
        gts = [frozenset({SpanLabel.LOCATION}), frozenset({SpanLabel.WORKPLACE})]
        assert compute_false_negative_rate(preds, gts) == 0.5

    def test_no_pii_examples_returns_zero(self):
        """ENSURES: 0.0 if no PII examples."""
        preds = [frozenset()]
        gts = [frozenset()]
        assert compute_false_negative_rate(preds, gts) == 0.0

    def test_clean_predictions_on_clean_text_not_counted(self):
        """Clean text with clean prediction is not a false negative."""
        preds = [frozenset(), frozenset({SpanLabel.LOCATION})]
        gts = [frozenset(), frozenset({SpanLabel.LOCATION})]
        assert compute_false_negative_rate(preds, gts) == 0.0


# ── §6: Hard Negative Precision ───────────────────────────────────────────


class TestHardNegativePrecision:
    """ENSURES: correct / count(hard_negatives)."""

    def _make_example(self, is_hard_negative, split="test"):
        return Example(
            id=f"{split}-00001",
            text="The Battle of Gettysburg",
            labels=frozenset(),
            risk=RiskLevel.LOW,
            rationale="",
            is_hard_negative=is_hard_negative,
            split=split,
            domain="personal",
            source="hard-negative" if is_hard_negative else "template",
        )

    def test_all_correct(self):
        examples = [self._make_example(True)]
        predictions = [DetectionResult(labels=frozenset(), risk=RiskLevel.LOW, rationale="")]
        assert compute_hard_negative_precision(examples, predictions) == 1.0

    def test_all_wrong(self):
        examples = [self._make_example(True)]
        wrong_pred = DetectionResult(
            labels=frozenset({SpanLabel.LOCATION}),
            risk=RiskLevel.MEDIUM,
            rationale="",
        )
        assert compute_hard_negative_precision(examples, [wrong_pred]) == 0.0

    def test_no_hard_negatives_returns_one(self):
        """ENSURES: 1.0 if no hard negative examples."""
        examples = [self._make_example(False)]
        predictions = [DetectionResult(labels=frozenset(), risk=RiskLevel.LOW, rationale="")]
        assert compute_hard_negative_precision(examples, predictions) == 1.0

    def test_non_hard_negatives_ignored(self):
        hn = self._make_example(True)
        non_hn = self._make_example(False)
        non_hn_pred = DetectionResult(
            labels=frozenset({SpanLabel.LOCATION}),
            risk=RiskLevel.MEDIUM,
            rationale="",
        )
        hn_pred = DetectionResult(labels=frozenset(), risk=RiskLevel.LOW, rationale="")
        assert compute_hard_negative_precision([hn, non_hn], [hn_pred, non_hn_pred]) == 1.0


# ── §2: Multilabel F1 ──────────────────────────────────────────────────────


class TestMultilabelF1:
    """ENSURES: macro-average across all 8 SpanLabels."""

    def test_perfect_single_label(self):
        """All predictions correct for one label → F1 for that label = 1.0."""
        preds = [frozenset({SpanLabel.LOCATION})]
        gts = [frozenset({SpanLabel.LOCATION})]
        agg, by_label = compute_multilabel_f1(preds, gts)
        assert by_label[SpanLabel.LOCATION] == 1.0

    def test_no_instances_label_gets_zero(self):
        """ENSURES: label with zero instances gets F1 = 0.0."""
        preds = [frozenset()]
        gts = [frozenset()]
        agg, by_label = compute_multilabel_f1(preds, gts)
        for label in SpanLabel:
            assert by_label[label] == 0.0
        assert agg == 0.0

    def test_macro_average_divides_by_eight(self):
        """ENSURES: macro-average counts all 8 labels in denominator."""
        preds = [frozenset({SpanLabel.LOCATION})]
        gts = [frozenset({SpanLabel.LOCATION})]
        agg, by_label = compute_multilabel_f1(preds, gts)
        assert abs(agg - 1.0 / 8.0) < 1e-9

    def test_by_label_has_eight_entries(self):
        preds = [frozenset()]
        gts = [frozenset()]
        _, by_label = compute_multilabel_f1(preds, gts)
        assert len(by_label) == 8

    def test_false_positive_label(self):
        """Predicting a label not in ground truth → precision drops."""
        preds = [frozenset({SpanLabel.LOCATION, SpanLabel.WORKPLACE})]
        gts = [frozenset({SpanLabel.LOCATION})]
        _, by_label = compute_multilabel_f1(preds, gts)
        assert by_label[SpanLabel.LOCATION] == 1.0
        assert by_label[SpanLabel.WORKPLACE] == 0.0

    def test_false_negative_label(self):
        """Label in ground truth but not predicted → recall drops."""
        preds = [frozenset({SpanLabel.LOCATION})]
        gts = [frozenset({SpanLabel.LOCATION, SpanLabel.WORKPLACE})]
        _, by_label = compute_multilabel_f1(preds, gts)
        assert by_label[SpanLabel.LOCATION] == 1.0
        assert by_label[SpanLabel.WORKPLACE] == 0.0  # 0 TP, 0 FP, 1 FN

    def test_zero_precision_denominator(self):
        """Label with no predictions but present in GT → precision=0, recall=0, F1=0."""
        preds = [frozenset()]
        gts = [frozenset({SpanLabel.ROUTINE})]
        _, by_label = compute_multilabel_f1(preds, gts)
        assert by_label[SpanLabel.ROUTINE] == 0.0

    def test_zero_recall_denominator(self):
        """Label predicted but never in GT → precision=0 (no TP), recall undefined → F1=0."""
        preds = [frozenset({SpanLabel.ROUTINE})]
        gts = [frozenset()]
        _, by_label = compute_multilabel_f1(preds, gts)
        assert by_label[SpanLabel.ROUTINE] == 0.0


# ── §5: Quasi-ID F1 ──────────────────────────────────────────────────────


class TestQuasiIdF1:
    """ENSURES: F1 on QUASI-ID label only."""

    def test_perfect_quasi_id(self):
        preds = [frozenset({SpanLabel.QUASI_ID})]
        gts = [frozenset({SpanLabel.QUASI_ID})]
        assert compute_quasi_id_f1(preds, gts) == 1.0

    def test_no_quasi_id_returns_zero(self):
        """ENSURES: 0.0 if no QUASI-ID in ground truth or predictions."""
        preds = [frozenset({SpanLabel.LOCATION})]
        gts = [frozenset({SpanLabel.LOCATION})]
        assert compute_quasi_id_f1(preds, gts) == 0.0

    def test_missed_quasi_id(self):
        """QUASI-ID in ground truth but not predicted → recall = 0."""
        preds = [frozenset({SpanLabel.LOCATION})]
        gts = [frozenset({SpanLabel.LOCATION, SpanLabel.QUASI_ID})]
        assert compute_quasi_id_f1(preds, gts) == 0.0

    def test_false_positive_quasi_id(self):
        """QUASI-ID predicted but not in ground truth → precision = 0."""
        preds = [frozenset({SpanLabel.QUASI_ID})]
        gts = [frozenset({SpanLabel.LOCATION})]
        assert compute_quasi_id_f1(preds, gts) == 0.0

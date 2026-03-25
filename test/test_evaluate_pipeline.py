"""Tests for the evaluation pipeline orchestrator.

All expectations derived from specifications/evaluation.md §1, §7.
"""

from __future__ import annotations

from contextual_pii_tagger.entities import (
    DetectionResult,
    RiskLevel,
    SpanLabel,
)
from contextual_pii_tagger.eval.evaluate import compare_models, evaluate
from contextual_pii_tagger.example import EvaluationReport, Example


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_example(
    text="I work at Acme Corp near downtown.",
    labels=None,
    risk=RiskLevel.MEDIUM,
    rationale="Combined risk.",
    is_hard_negative=False,
    idx=1,
):
    if labels is None:
        labels = frozenset({SpanLabel.WORKPLACE, SpanLabel.LOCATION})
    return Example(
        id=f"test-{idx:05d}",
        text=text,
        labels=labels,
        risk=risk,
        rationale=rationale,
        is_hard_negative=is_hard_negative,
        split="test",
        domain="workplace",
        source="hard-negative" if is_hard_negative else "template",
    )


def _make_hard_negative(idx=2):
    return _make_example(
        text="The Battle of Gettysburg was important.",
        labels=frozenset(),
        risk=RiskLevel.LOW,
        rationale="",
        is_hard_negative=True,
        idx=idx,
    )


class _FakePredictor:
    """Returns fixed predictions for testing."""

    def __init__(self, results: list[DetectionResult]):
        self._results = results
        self._idx = 0

    def predict(self, text: str) -> DetectionResult:
        result = self._results[self._idx]
        self._idx += 1
        return result


def _perfect_predictor(examples: list[Example]) -> _FakePredictor:
    """Predictor that returns ground-truth labels."""
    results = []
    for ex in examples:
        results.append(
            DetectionResult(
                labels=ex.labels,
                risk=ex.risk,
                rationale=ex.rationale,
            )
        )
    return _FakePredictor(results)


# ── §1: evaluate ────────────────────────────────────────────────────────


class TestEvaluate:
    """ENSURES: EvaluationReport from model + test dataset."""

    def test_returns_evaluation_report(self):
        examples = [_make_example(idx=1), _make_hard_negative(idx=2)]
        predictor = _perfect_predictor(examples)
        report = evaluate(predictor, examples, model_name="test-model")
        assert isinstance(report, EvaluationReport)

    def test_perfect_predictor_gets_full_marks(self):
        examples = [_make_example(idx=1), _make_hard_negative(idx=2)]
        predictor = _perfect_predictor(examples)
        report = evaluate(predictor, examples, model_name="test-model")
        assert report.risk_accuracy == 1.0
        assert report.false_negative_rate == 0.0
        assert report.hard_negative_precision == 1.0

    def test_report_has_correct_test_set_size(self):
        examples = [_make_example(idx=i) for i in range(1, 4)] + [
            _make_hard_negative(idx=4)
        ]
        predictor = _perfect_predictor(examples)
        report = evaluate(predictor, examples, model_name="test-model")
        assert report.test_set_size == 4

    def test_report_has_per_label_f1(self):
        examples = [_make_example(idx=1), _make_hard_negative(idx=2)]
        predictor = _perfect_predictor(examples)
        report = evaluate(predictor, examples, model_name="test-model")
        assert len(report.f1_by_label) == 8

    def test_deterministic(self):
        """ENSURES: same inputs → same report."""
        examples = [_make_example(idx=1), _make_hard_negative(idx=2)]
        p1 = _perfect_predictor(examples)
        r1 = evaluate(p1, examples, model_name="m")
        p2 = _perfect_predictor(examples)
        r2 = evaluate(p2, examples, model_name="m")
        assert r1 == r2

    def test_all_wrong_predictor(self):
        examples = [_make_example(idx=1)]
        wrong = DetectionResult(labels=frozenset(), risk=RiskLevel.LOW, rationale="")
        predictor = _FakePredictor([wrong])
        report = evaluate(predictor, examples, model_name="bad")
        assert report.false_negative_rate == 1.0
        assert report.risk_accuracy == 0.0


# ── §7: compare_models ─────────────────────────────────────────────────


class TestCompareModels:
    """ENSURES: side-by-side metrics with deltas."""

    def _make_report(self, name, f1=0.5, risk_acc=0.5, fnr=0.1, qi_f1=0.5, hnp=0.9):
        by_label = {label: f1 for label in SpanLabel}
        return EvaluationReport(
            model_name=name,
            test_set_size=100,
            multilabel_f1=f1,
            f1_by_label=by_label,
            risk_accuracy=risk_acc,
            false_negative_rate=fnr,
            quasi_id_f1=qi_f1,
            hard_negative_precision=hnp,
        )

    def test_returns_comparison_summary(self):
        ft = self._make_report("finetuned", f1=0.8, risk_acc=0.9)
        bl = self._make_report("baseline", f1=0.5, risk_acc=0.6)
        summary = compare_models(ft, bl)
        assert "finetuned" in summary
        assert "baseline" in summary

    def test_deltas_are_finetuned_minus_baseline(self):
        ft = self._make_report("finetuned", f1=0.8, risk_acc=0.9)
        bl = self._make_report("baseline", f1=0.5, risk_acc=0.6)
        summary = compare_models(ft, bl)
        assert abs(summary["deltas"]["multilabel_f1"] - 0.3) < 1e-6
        assert abs(summary["deltas"]["risk_accuracy"] - 0.3) < 1e-6

    def test_per_label_comparison(self):
        ft = self._make_report("finetuned", f1=0.8)
        bl = self._make_report("baseline", f1=0.5)
        summary = compare_models(ft, bl)
        assert "f1_by_label" in summary
        for label in SpanLabel:
            assert label.value in summary["f1_by_label"]

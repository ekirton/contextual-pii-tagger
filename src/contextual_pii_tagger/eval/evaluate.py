"""Evaluation pipeline orchestrator.

Spec: specifications/evaluation.md §1, §7
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

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
from contextual_pii_tagger.example import EvaluationReport, Example

logger = logging.getLogger(__name__)


class Predictor(Protocol):
    def predict(self, text: str) -> DetectionResult: ...


def evaluate(
    model: Predictor,
    test_dataset: list[Example],
    *,
    model_name: str = "model",
) -> EvaluationReport:
    """Compute all metrics for *model* on *test_dataset*.

    ENSURES:
        - Returns a valid EvaluationReport.
        - Deterministic: same inputs → same report.
    """
    predictions: list[DetectionResult] = []
    for ex in test_dataset:
        predictions.append(model.predict(ex.text))

    pred_labels = [p.labels for p in predictions]
    gt_labels = [ex.labels for ex in test_dataset]

    pred_risks = [p.risk for p in predictions]
    gt_risks = [ex.risk for ex in test_dataset]

    # Compute all 5 metrics
    multilabel_f1, f1_by_label = compute_multilabel_f1(pred_labels, gt_labels)
    risk_accuracy = compute_risk_accuracy(pred_risks, gt_risks)
    false_negative_rate = compute_false_negative_rate(pred_labels, gt_labels)
    quasi_id_f1 = compute_quasi_id_f1(pred_labels, gt_labels)
    hard_negative_precision = compute_hard_negative_precision(
        test_dataset, predictions
    )

    return EvaluationReport(
        model_name=model_name,
        test_set_size=len(test_dataset),
        multilabel_f1=multilabel_f1,
        f1_by_label=f1_by_label,
        risk_accuracy=risk_accuracy,
        false_negative_rate=false_negative_rate,
        quasi_id_f1=quasi_id_f1,
        hard_negative_precision=hard_negative_precision,
    )


def compare_models(
    finetuned_report: EvaluationReport,
    baseline_report: EvaluationReport,
) -> dict[str, Any]:
    """Produce a side-by-side comparison of two models.

    REQUIRES:
        - Both reports computed on the same test dataset.

    ENSURES:
        - Returns a ComparisonSummary dict with deltas = finetuned - baseline.
    """
    metric_keys = [
        "multilabel_f1",
        "risk_accuracy",
        "false_negative_rate",
        "quasi_id_f1",
        "hard_negative_precision",
    ]

    finetuned_metrics = {k: getattr(finetuned_report, k) for k in metric_keys}
    baseline_metrics = {k: getattr(baseline_report, k) for k in metric_keys}
    deltas = {k: finetuned_metrics[k] - baseline_metrics[k] for k in metric_keys}

    # Per-label F1 comparison
    f1_by_label: dict[str, dict[str, float]] = {}
    for label in SpanLabel:
        f1_by_label[label.value] = {
            "finetuned": finetuned_report.f1_by_label[label],
            "baseline": baseline_report.f1_by_label[label],
            "delta": (
                finetuned_report.f1_by_label[label]
                - baseline_report.f1_by_label[label]
            ),
        }

    return {
        "finetuned": finetuned_metrics,
        "baseline": baseline_metrics,
        "deltas": deltas,
        "f1_by_label": f1_by_label,
        "test_set_size": finetuned_report.test_set_size,
    }

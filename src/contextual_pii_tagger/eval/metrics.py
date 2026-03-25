"""Evaluation metrics for multilabel classification.

Spec: specifications/evaluation.md §2-6
"""

from __future__ import annotations

from contextual_pii_tagger.entities import (
    DetectionResult,
    RiskLevel,
    SpanLabel,
)
from contextual_pii_tagger.example import Example


def compute_multilabel_f1(
    predictions: list[frozenset[SpanLabel]],
    ground_truths: list[frozenset[SpanLabel]],
) -> tuple[float, dict[SpanLabel, float]]:
    """Compute per-label and macro-averaged multilabel F1.

    Macro-average across all 8 SpanLabels. Labels with zero instances get F1=0.0.
    """
    label_tp: dict[SpanLabel, int] = {label: 0 for label in SpanLabel}
    label_fp: dict[SpanLabel, int] = {label: 0 for label in SpanLabel}
    label_fn: dict[SpanLabel, int] = {label: 0 for label in SpanLabel}

    for pred_set, gt_set in zip(predictions, ground_truths):
        for label in SpanLabel:
            in_pred = label in pred_set
            in_gt = label in gt_set
            if in_pred and in_gt:
                label_tp[label] += 1
            elif in_pred and not in_gt:
                label_fp[label] += 1
            elif not in_pred and in_gt:
                label_fn[label] += 1

    by_label: dict[SpanLabel, float] = {}
    for label in SpanLabel:
        tp = label_tp[label]
        fp = label_fp[label]
        fn = label_fn[label]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            by_label[label] = 2 * precision * recall / (precision + recall)
        else:
            by_label[label] = 0.0

    aggregate = sum(by_label.values()) / len(by_label)
    return aggregate, by_label


def compute_risk_accuracy(
    predictions: list[RiskLevel], ground_truths: list[RiskLevel]
) -> float:
    """Fraction of examples where predicted risk matches ground truth."""
    if not predictions:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    return correct / len(predictions)


def compute_false_negative_rate(
    predictions: list[frozenset[SpanLabel]],
    ground_truths: list[frozenset[SpanLabel]],
) -> float:
    """Fraction of PII-containing texts classified as clean (empty label set)."""
    pii_examples = 0
    missed = 0
    for pred_set, gt_set in zip(predictions, ground_truths):
        if gt_set:  # ground truth has PII
            pii_examples += 1
            if not pred_set:  # predicted as clean
                missed += 1
    if pii_examples == 0:
        return 0.0
    return missed / pii_examples


def compute_quasi_id_f1(
    predictions: list[frozenset[SpanLabel]],
    ground_truths: list[frozenset[SpanLabel]],
) -> float:
    """F1 computed only on the QUASI-ID label across all examples."""
    tp = 0
    fp = 0
    fn = 0
    for pred_set, gt_set in zip(predictions, ground_truths):
        in_pred = SpanLabel.QUASI_ID in pred_set
        in_gt = SpanLabel.QUASI_ID in gt_set
        if in_pred and in_gt:
            tp += 1
        elif in_pred and not in_gt:
            fp += 1
        elif not in_pred and in_gt:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def compute_hard_negative_precision(
    examples: list[Example], predictions: list[DetectionResult]
) -> float:
    """Fraction of hard negatives correctly predicted as clean."""
    hard_neg_count = 0
    correct = 0
    for ex, pred in zip(examples, predictions):
        if ex.is_hard_negative:
            hard_neg_count += 1
            if not pred.labels:
                correct += 1
    if hard_neg_count == 0:
        return 1.0
    return correct / hard_neg_count

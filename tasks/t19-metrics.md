# T-19: Evaluation Metrics

**File:** `src/contextual_pii_tagger/eval/metrics.py`
**Spec:** [evaluation.md](../specifications/evaluation.md) §2-6
**Depends on:** T-01 (entities.py), T-02 (example.py)

## Scope

Implement all five metric computation functions for multilabel classification evaluation.

## Deliverables

1. **compute_multilabel_f1(predictions, ground_truths)** — Per-label precision/recall/F1 computed across all examples by comparing predicted and ground-truth label sets. Macro-averaged across all 8 SpanLabels. Return aggregate F1 + per-label map.

2. **compute_risk_accuracy(predictions, ground_truths)** — Fraction of exact RiskLevel matches.

3. **compute_false_negative_rate(predictions, ground_truths)** — Fraction of PII-containing texts (non-empty ground-truth labels) where predicted label set is empty.

4. **compute_quasi_id_f1(predictions, ground_truths)** — F1 on the QUASI-ID label only across all examples.

5. **compute_hard_negative_precision(examples, predictions)** — Fraction of hard negatives with empty predicted labels.

## Acceptance Criteria

- All functions return float in [0.0, 1.0].
- Multilabel F1 is macro-average (not micro).
- Zero-division cases handled per spec (return 0.0 or 1.0 as specified).
- Labels with zero instances still counted in macro-average denominator.

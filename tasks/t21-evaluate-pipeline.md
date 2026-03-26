# T-21: Evaluation Pipeline Orchestrator

**File:** `src/contextual_pii_tagger/eval/evaluate.py`
**Spec:** [evaluation.md](../specifications/evaluation.md) §1, §7
**Depends on:** T-19 (metrics.py), T-20 (baseline.py), T-06 (detector.py)

## Scope

Implement `evaluate(model, test_dataset) -> EvaluationReport` and `compare_models`.

## Deliverables

1. **evaluate function** — For each test Example:
   - Run model.predict(text) to get DetectionResult
   - Collect predicted label sets and risk levels
   - Compute all 5 metrics (multilabel F1, risk accuracy, false negative rate, quasi-ID F1, hard negative precision)
   - Build EvaluationReport

2. **compare_models(finetuned_report, baseline_report)** — Produce ComparisonSummary:
   - Side-by-side metrics with deltas
   - Per-SpanLabel F1 comparison
   - RiskLevel confusion matrices
   - Binary gate accuracy (PII-present vs. clean)
   - Disagreement Example IDs

3. **CLI entry point** — Accept model path, test set path, optional baseline flag. Output reports as JSON.

## Acceptance Criteria

- EvaluationReport satisfies all entity invariants.
- Results deterministic.
- Comparison deltas = finetuned - baseline.
- Confusion matrix correct for each model.

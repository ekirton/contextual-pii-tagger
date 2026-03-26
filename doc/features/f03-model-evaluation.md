# F-03: Model Evaluation

**Priority:** Core

## What This Feature Does

Measures the model's detection quality against the success metrics and compares performance against an XGBoost baseline trained on the same dataset. Produces a structured evaluation report showing how the model performs on each metric and where it outperforms or underperforms the baseline.

The evaluation covers:

- **Multilabel F1** — whether the correct set of quasi-identifier categories is identified (target: >= 0.80)
- **Risk score accuracy** — whether the overall risk classification is correct (target: >= 0.85)
- **False negative rate** — how often text containing PII is classified as clean (target: <= 0.08)
- **Quasi-ID F1** — detection quality on the hardest category, quasi-identifier combinations (target: >= 0.70)
- **Hard negative precision** — how often non-PII text is correctly classified as clean (target: >= 0.92)

## Why It Exists

Without rigorous evaluation, there is no way to know whether the model actually solves the problem it claims to solve. The XGBoost baseline isolates the value of contextual reasoning — if a traditional classifier using handcrafted features can match the model's performance, the approach is not justified.

## Design Tradeoffs

- The evaluation uses the project's own test set rather than an external benchmark, because no external benchmark for Tier 2 quasi-identifier detection exists. This means the evaluation measures in-distribution performance, not generalization to unseen domains.
- The XGBoost baseline is the only comparison. Existing PII tools (Presidio, spaCy) are not meaningful baselines for Tier 2 since they have near-zero recall on quasi-identifiers by design.
- Metric targets are aspirational for a proof of concept. The primary goal is to demonstrate that contextual reasoning adds value over traditional features, not to achieve production-grade accuracy.

## What This Feature Does Not Provide

- Evaluation on non-English text or adversarial inputs.
- Comparison against Tier 1 detection tools (these solve a different problem).
- Cross-domain generalization testing.
- Latency or throughput benchmarks.

## Acceptance Criteria

### AC-01: All success metrics reported
**GIVEN** the evaluation has been run on the test set
**WHEN** the evaluation report is produced
**THEN** it includes values for all five success metrics: Multilabel F1, Risk score accuracy, False negative rate, Quasi-ID F1, and Hard negative precision

### AC-02: XGBoost baseline comparison
**GIVEN** an XGBoost classifier trained on the same dataset with handcrafted features
**WHEN** both models are evaluated on the same test set
**THEN** the report shows side-by-side results for the fine-tuned model and the XGBoost baseline across all metrics

### AC-03: Per-category breakdown
**GIVEN** the evaluation has completed
**WHEN** the report is reviewed
**THEN** detection performance is broken down by individual Tier 2 category (LOCATION, WORKPLACE, ROUTINE, etc.), not only reported as an aggregate

### AC-04: Reproducibility
**GIVEN** the evaluation script and test set
**WHEN** the evaluation is re-run
**THEN** results are deterministic and reproducible

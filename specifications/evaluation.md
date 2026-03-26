# Evaluation Specification

**Architecture:** [evaluation-pipeline.md](../doc/architecture/evaluation-pipeline.md)

---

## 1. evaluate

```
evaluate(model: Predictor, test_dataset: list<Example>) -> EvaluationReport
```

Top-level function that computes all metrics for a single model.

**REQUIRES:**
- `model` implements a `predict(text: string) -> DetectionResult` interface (either the fine-tuned PIIDetector or the XGBoost baseline wrapper).
- `test_dataset` is a list of valid Example records from the test split.
- `len(test_dataset)` == 5,000.

**ENSURES:**
- Returns a valid EvaluationReport (satisfies all invariants from entities.md Section 5).
- All metrics are computed on the full test dataset.
- Results are deterministic: calling `evaluate` twice with the same inputs produces the same report.

---

## 2. Multilabel F1

### 2.1 compute_multilabel_f1

```
compute_multilabel_f1(predictions: list<set<SpanLabel>>, ground_truths: list<set<SpanLabel>>) -> (float, map<SpanLabel, float>)
```

**REQUIRES:**
- `len(predictions) == len(ground_truths)`.
- All values in each set are valid SpanLabel values.

**ENSURES:**
- For each SpanLabel, computes across all examples:
  - `true_positives` = count of examples where the label is in both predicted and ground-truth sets.
  - `false_positives` = count of examples where the label is in predicted but not ground-truth.
  - `false_negatives` = count of examples where the label is in ground-truth but not predicted.
  - `precision` = true_positives / (true_positives + false_positives), or 0.0 if denominator is 0.
  - `recall` = true_positives / (true_positives + false_negatives), or 0.0 if denominator is 0.
  - `f1` = 2 * precision * recall / (precision + recall), or 0.0 if both are 0.
- The aggregate `multilabel_f1` is the **macro-average** of F1 across all 8 SpanLabel values.
- If a SpanLabel has zero ground-truth and zero predicted instances, its F1 is 0.0 (it is still counted as one of the 8 labels in the macro-average denominator).
- Returns the aggregate F1 and a map of per-label F1 values.

---

## 3. Risk Score Accuracy

### 3.1 compute_risk_accuracy

```
compute_risk_accuracy(predictions: list<RiskLevel>, ground_truths: list<RiskLevel>) -> float
```

**REQUIRES:**
- `len(predictions) == len(ground_truths)`.
- All values are valid RiskLevel.

**ENSURES:**
- Returns the fraction of examples where `predictions[i] == ground_truths[i]`.
- Range: [0.0, 1.0].

---

## 4. False Negative Rate

### 4.1 compute_false_negative_rate

```
compute_false_negative_rate(predictions: list<set<SpanLabel>>, ground_truths: list<set<SpanLabel>>) -> float
```

**REQUIRES:**
- `len(predictions) == len(ground_truths)`.

**ENSURES:**
- `pii_examples` = count of examples where ground-truth label set is non-empty.
- `missed` = count of examples where ground-truth label set is non-empty but predicted label set is empty.
- Returns `missed / pii_examples`, or 0.0 if `pii_examples` is 0.
- Range: [0.0, 1.0].

---

## 5. Quasi-ID F1

### 5.1 compute_quasi_id_f1

```
compute_quasi_id_f1(predictions: list<set<SpanLabel>>, ground_truths: list<set<SpanLabel>>) -> float
```

**REQUIRES:**
- `len(predictions) == len(ground_truths)`.

**ENSURES:**
- Considers only the QUASI-ID label across all examples.
- `true_positives` = count of examples where QUASI-ID is in both predicted and ground-truth sets.
- `false_positives` = count of examples where QUASI-ID is in predicted but not ground-truth.
- `false_negatives` = count of examples where QUASI-ID is in ground-truth but not predicted.
- Computes precision, recall, and F1 on these counts.
- Returns F1, or 0.0 if there are no QUASI-ID ground-truth or predicted instances.

---

## 6. Hard Negative Precision

### 6.1 compute_hard_negative_precision

```
compute_hard_negative_precision(examples: list<Example>, predictions: list<DetectionResult>) -> float
```

**REQUIRES:**
- `len(examples) == len(predictions)`.
- Each Example has a valid `is_hard_negative` field.

**ENSURES:**
- `hard_neg_examples` = examples where `is_hard_negative == true`.
- `correct` = count of hard negative examples where the corresponding prediction has empty labels.
- Returns `correct / len(hard_neg_examples)`, or 1.0 if there are no hard negative examples.
- Range: [0.0, 1.0].

---

## 7. Comparison Report

### 7.1 compare_models

```
compare_models(finetuned_report: EvaluationReport, baseline_report: EvaluationReport) -> ComparisonSummary
```

**REQUIRES:**
- Both reports were computed on the same test dataset.
- `finetuned_report.test_set_size == baseline_report.test_set_size`.

**ENSURES:**
- Returns a ComparisonSummary containing:
  - Side-by-side values for all metrics.
  - Per-SpanLabel F1 comparison.
  - Delta (finetuned - baseline) for each metric.
  - A confusion matrix for RiskLevel predictions for each model.
  - Binary gate accuracy (fraction of examples correctly classified as PII-present vs. clean) for each model.
  - List of Example IDs where the two models' predictions disagree on risk level.

---

## 8. XGBoost Baseline

### 8.1 train_baseline

```
train_baseline(train_dataset: list<Example>) -> XGBoostPredictor
```

**REQUIRES:**
- `train_dataset` is the training split (40,000 examples).

**ENSURES:**
- Returns an XGBoostPredictor that implements the `predict(text: string) -> DetectionResult` interface.
- Features are extracted per evaluation-pipeline.md Section 3: TF-IDF, entity counts, token patterns, text statistics.
- The predictor performs two classification tasks:
  1. Multilabel classification (one binary classifier per SpanLabel, thresholded at 0.5).
  2. Risk classification (RiskLevel for the full text).
- The `rationale` field in predicted DetectionResults is always an empty string.
- Hyperparameters are tuned via cross-validation on the training set only.

### 8.2 extract_features

```
extract_features(text: string) -> FeatureVector
```

**REQUIRES:**
- `text` is a non-empty string.

**ENSURES:**
- Returns a fixed-dimension feature vector containing:
  - TF-IDF features over the training vocabulary.
  - Named entity counts from spaCy: count of PERSON, ORG, GPE, DATE entities.
  - Token pattern indicators: presence of email-like, phone-like, address-like, credential-like patterns.
  - Text statistics: token count, sentence count, average sentence length.
- The feature extraction is deterministic.

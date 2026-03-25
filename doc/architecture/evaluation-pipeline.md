# Evaluation Pipeline

**Features:** F-03 (Model Evaluation)
**Requirements:** R-EVL-01, R-EVL-02

---

## 1. Pipeline Overview

The evaluation pipeline measures detection quality for both the fine-tuned model and the XGBoost baseline, producing a side-by-side EvaluationReport.

```
                          ┌──────────────────┐
                          │    Test Split     │
                          │ (5,000 Examples)  │
                          └────────┬─────────┘
                                   │
                          ┌────────┴─────────┐
                          │                  │
                   ┌──────v──────┐    ┌──────v──────┐
                   │  Fine-Tuned │    │   XGBoost   │
                   │    Model    │    │  Baseline   │
                   │ (Inference) │    │ (Predict)   │
                   └──────┬──────┘    └──────┬──────┘
                          │                  │
                   ┌──────v──────┐    ┌──────v──────┐
                   │  Predicted  │    │  Predicted  │
                   │  Results    │    │  Results    │
                   └──────┬──────┘    └──────┬──────┘
                          │                  │
                          └────────┬─────────┘
                                   │
                          ┌────────v─────────┐
                          │  Metric Compute  │
                          │  & Comparison    │
                          └────────┬─────────┘
                                   │
                          ┌────────v─────────┐
                          │ EvaluationReport │
                          └──────────────────┘
```

## 2. Fine-Tuned Model Evaluation

For each Example in the test split, the Detection Interface runs inference and produces a predicted DetectionResult. The predicted label set and RiskLevel are compared against the ground-truth labels and RiskLevel in the Example record.

## 3. XGBoost Baseline

The XGBoost baseline is a traditional machine learning classifier trained on the same training split, using handcrafted features rather than language model reasoning.

### Feature Extraction

Each Example's text is transformed into a fixed-dimension feature vector:

| Feature Group | Description |
|---------------|-------------|
| TF-IDF | Term frequency-inverse document frequency vectors over the full vocabulary |
| Entity counts | Count of named entities detected by spaCy (persons, organizations, locations, dates) |
| Token patterns | Presence of specific token patterns (email-like, phone-like, address-like, credential-like) |
| Text statistics | Token count, sentence count, average sentence length |

### Task Formulation

The XGBoost baseline performs two classification tasks:

1. **Multilabel classification:** Given the full text and its extracted features, predict which SpanLabel categories are present. One binary classifier per SpanLabel, thresholded at 0.5.
2. **Risk classification:** Given the full text and its extracted features, classify the RiskLevel as LOW, MEDIUM, or HIGH.

The baseline does not produce a rationale — this field is left empty in its predicted results.

### Training

The XGBoost classifier is trained on the same 40,000 training examples using the extracted features. Hyperparameters are tuned via cross-validation on the training set (not the validation split, which is reserved for the fine-tuned model's training monitoring).

## 4. Metrics

All metrics are computed on the test split (5,000 Examples) for both models.

### 4.1 Multilabel F1

For each SpanLabel, compute precision, recall, and F1 by comparing the predicted label set against the ground-truth label set across all test Examples. A label is a true positive for an Example if it appears in both the predicted and ground-truth sets.

The aggregate Multilabel F1 is the macro-average across all SpanLabel values.

**Target:** >= 0.80

### 4.2 Risk Score Accuracy

Fraction of test Examples where the predicted RiskLevel exactly matches the ground-truth RiskLevel.

**Target:** >= 0.85

### 4.3 False Negative Rate

Fraction of test Examples where the ground-truth label set is non-empty but the predicted label set is empty (i.e., PII-containing text classified as clean).

**Target:** <= 0.08

### 4.4 Quasi-ID F1

Precision, recall, and F1 computed only on the QUASI-ID label across all test Examples. This isolates performance on the hardest detection category — text where multiple details are only sensitive in aggregate.

**Target:** >= 0.70

### 4.5 Hard Negative Precision

Among all test Examples where `is_hard_negative: true`, the fraction where the model correctly returns an empty label set.

**Target:** >= 0.92

## 5. Output: EvaluationReport

The pipeline produces an EvaluationReport (as defined in system-overview.md Section 3.6) for each model, plus a comparison summary showing:

- Side-by-side metric values for the fine-tuned model and XGBoost baseline
- Per-SpanLabel F1 breakdown for both models
- Confusion matrix for RiskLevel predictions
- Binary gate accuracy (PII present vs. clean) for both models
- List of Example IDs where the models disagree (for qualitative analysis)


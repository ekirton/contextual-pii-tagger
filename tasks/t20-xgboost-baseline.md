# T-20: XGBoost Baseline

**File:** `src/contextual_pii_tagger/eval/baseline.py`
**Spec:** [evaluation.md](../specifications/evaluation.md) §8
**Depends on:** T-01 (entities.py), T-02 (example.py)

## Scope

Implement `train_baseline(train_dataset) -> XGBoostPredictor` and `extract_features(text) -> FeatureVector`.

## Deliverables

1. **extract_features** — Compute feature vector:
   - TF-IDF over training vocabulary
   - spaCy entity counts (PERSON, ORG, GPE, DATE)
   - Token pattern indicators (email-like, phone-like, address-like, credential-like)
   - Text statistics (token count, sentence count, avg sentence length)

2. **Multilabel classifier** — One binary XGBoost classifier per SpanLabel, thresholded at 0.5.

3. **Risk classifier** — XGBoost multi-class: RiskLevel for full text.

4. **XGBoostPredictor** — Wrapper implementing `predict(text) -> DetectionResult`. Rationale always empty string.

5. **Hyperparameter tuning** — Cross-validation on training set only.

## Acceptance Criteria

- Predictor implements same interface as PIIDetector.detect.
- Feature extraction is deterministic.
- Rationale always empty in predictions.
- No use of validation or test data during training.

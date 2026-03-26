# F-06: Human Review Workflow

**Priority:** Core

## What This Feature Does

Provides a workflow for human annotators to spot-check a random 1% sample of generated examples across all dataset splits. Annotators review the category labels, risk scores, and hard negative classifications, correcting any errors before the dataset is finalized.

## Why It Exists

Synthetic data generation is imperfect. Category labels may be wrong or missing, and hard negatives may actually contain PII. Human spot-checking catches systematic errors in the generation pipeline and ensures the dataset meets a minimum quality bar — especially for the validation and test sets that the model is evaluated against.

## Design Tradeoffs

- Only 1% of examples are reviewed rather than the full dataset. This balances quality assurance against the cost and time of manual annotation. If the 1% sample reveals high error rates, the generation pipeline should be improved rather than expanding review coverage.
- The review covers all splits uniformly (train, validation, test) rather than prioritizing the test set. This ensures that training data quality issues are caught, not just evaluation data issues.

## What This Feature Does Not Provide

- Full manual labeling of the dataset. The majority of labels are auto-generated.
- Ongoing review after initial dataset creation.
- Inter-annotator agreement measurement or formal annotation guidelines (appropriate for a proof of concept but not for a production dataset).

## Acceptance Criteria

### AC-01: 1% sample reviewed
**GIVEN** the synthetic dataset has been generated
**WHEN** human review is complete
**THEN** at least 1% of examples across all splits have been manually reviewed and any errors corrected

### AC-02: Random selection
**GIVEN** the examples selected for review
**WHEN** the selection is inspected
**THEN** examples were chosen randomly, not cherry-picked by domain or difficulty

### AC-03: Error correction
**GIVEN** a reviewer identifies an incorrect category label, missing label, or misclassified hard negative
**WHEN** they correct it
**THEN** the correction is reflected in the final dataset

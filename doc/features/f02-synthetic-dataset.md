# F-02: Synthetic Dataset Generation

**Priority:** Core

## What This Feature Does

Produces the complete training dataset for the model — 20,000 synthetically generated examples of free text, each labeled with the quasi-identifier categories present and risk scores. The dataset is split 80:10:10 into train (16,000), validation (2,000), and test (2,000) partitions.

Each split includes 10% hard negatives: text that mentions places, times, organizations, or other details that look like quasi-identifiers but are not PII in context (e.g., historical references, fictional characters, public figures discussed in public contexts).

## Why It Exists

Training a PII detector on real PII is both legally problematic and ethically undesirable. Synthetic generation is the standard approach in this domain — it produces realistic, controllable, and fully labeled examples without handling actual personal information.

Hard negatives are essential because without them the model would learn to flag any mention of a location, workplace, or schedule as PII, producing excessive false alarms that erode user trust.

## Design Tradeoffs

- Synthetic data may not capture the full diversity of real-world prompts. This is partially mitigated by using multiple generation approaches (templates and LLM-augmented generation) and by including diverse domains (medical, scheduling, workplace, personal).
- 10% hard negatives is a starting ratio. If evaluation reveals excessive false positives, the ratio may need to increase — but this is a tuning decision, not a feature change.
- The 80:10:10 split is standard. The validation and test sets are large enough (2,000 each) to provide statistically meaningful evaluation results.

## What This Feature Does Not Provide

- Real-world PII examples. All data is synthetic.
- Tier 1 (direct identifier) training examples. The dataset focuses on Tier 2 quasi-identifiers.
- Non-English examples.
- Adversarially obfuscated examples.

## Acceptance Criteria

### AC-01: Dataset size and split
**GIVEN** the generation pipeline has completed
**WHEN** the dataset is inspected
**THEN** it contains 20,000 total examples split into 16,000 train, 2,000 validation, and 2,000 test

### AC-02: Hard negatives in every split
**GIVEN** any single split (train, validation, or test)
**WHEN** the split is inspected
**THEN** approximately 10% of examples are hard negatives — text that resembles PII but is not sensitive in context

### AC-03: No real personal data
**GIVEN** any example in the dataset
**WHEN** the example is inspected
**THEN** all names, locations, organizations, and other details are synthetically generated and do not correspond to real individuals

### AC-04: Category labels and risk scores
**GIVEN** any non-hard-negative example in the dataset
**WHEN** the example is inspected
**THEN** it includes the set of quasi-identifier categories present from the Tier 2 taxonomy and an overall risk score (LOW/MEDIUM/HIGH)

### AC-05: Domain diversity
**GIVEN** the complete dataset
**WHEN** examples are reviewed across the dataset
**THEN** they span multiple domains including medical context, scheduling, workplace, and personal introductions

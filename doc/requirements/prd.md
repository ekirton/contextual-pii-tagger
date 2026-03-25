# Product Requirements Document: contextual-pii-tagger

## 1. Product Goal

Detect personally identifiable information that existing tools miss — specifically, contextual and quasi-identifier PII where combinations of seemingly innocuous details (workplace, schedule, demographics, location) together re-identify or locate an individual.

The product is a proof of concept released as open-source on HuggingFace for community use and contribution.

## 2. Target User Segments

| Segment | Need |
|---------|------|
| AI application developers | Prevent accidental PII disclosure in prompts sent to external LLM APIs |
| Privacy and compliance teams | Audit and flag employee interactions containing customer or patient data |
| Data engineers | Screen documents for quasi-identifier PII before indexing or sharing |
| Open-source/research community | Reproduce, benchmark, and extend contextual PII detection |

## 3. Competitive Context

Existing PII detection tools — Microsoft Presidio, spaCy NER, cloud provider PII APIs — reliably detect direct identifiers: names, emails, phone numbers, government IDs, and similar structured data. None of these tools attempt to detect contextual PII, where the sensitivity arises from the combination of details rather than from any single field. This is an unsolved gap in the market.

## 4. Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Category label detection (multilabel F1) | >= 0.80 | Must reliably identify which quasi-identifier categories are present |
| Correct risk score classification | >= 0.85 | Users depend on risk level to decide whether to act |
| Missed PII rate (false negatives on binary gate) | <= 0.08 | Missing real PII undermines trust in the product |
| Quasi-identifier combination detection (F1 on QUASI-ID label) | >= 0.70 | Hardest category; the core differentiator |
| Non-PII correctly classified clean (hard negative precision) | >= 0.92 | Excessive false alarms erode user confidence |

## 5. Scope

### 5.1 In Scope

| ID | Requirement | Priority |
|----|-------------|----------|
| R-DET-01 | Classify which quasi-identifier categories are present in free text — location, workplace, routine, medical context, demographics, device identifiers, credentials, and quasi-identifier combinations — without requiring span-level extraction | P0 |
| R-DET-02 | Assign an overall risk score (LOW / MEDIUM / HIGH) to each analyzed text | P0 |
| R-DET-03 | Provide a brief rationale explaining why quasi-identifier combinations are sensitive | P0 |
| R-DET-04 | Operate entirely offline with no network calls during detection | P0 |
| R-DAT-01 | Produce a synthetically generated training dataset of 20,000 examples in an 80:10:10 train/validation/test split | P0 |
| R-DAT-02 | Include 10% hard negatives (text that resembles PII but is not) in each split | P0 |
| R-DAT-03 | Human-spot-check 1% of examples across all splits | P1 |
| R-EVL-01 | Evaluate the model against all success metrics defined in Section 4 | P0 |
| R-EVL-02 | Compare performance against a traditional machine learning baseline (XGBoost) trained on the same dataset | P0 |
| R-API-01 | Provide a detection interface usable from application code | P0 |
| R-DMO-01 | Demonstrate the detector as a Claude Code privacy hook that blocks PII before it reaches external APIs | P1 |
| R-DMO-02 | The hook presents findings conversationally, offering the user options to redact, revise, or continue | P1 |
| R-T1-01 | Detect Tier 1 direct identifiers (names, emails, phone numbers, government IDs, addresses, financial account numbers, dates of birth) using a third-party NER library — no custom Tier 1 model is trained | P1 |
| R-BIN-01 | Provide a single compiled binary that performs both Tier 1 direct identifier detection and Tier 2 quasi-identifier detection, replacing the Python inference and hook layer for runtime use | P1 |
| R-BIN-02 | The binary preserves the hook exit code contract (0 pass / 1 error / 2 block) and the PII_MODEL_PATH environment variable interface | P1 |
| R-ETH-01 | Use no real personal data in training — all examples must be synthetically generated | P0 |

### 5.2 Out of Scope

- **Custom Tier 1 model training.** Tier 1 direct identifier detection is handled by a third-party NER library (redact-core) bundled within the compiled binary. No custom Tier 1 model is trained or fine-tuned.
- **Sensitive context detection (Tier 3).** Medical diagnoses, legal history, financial distress, political affiliation, religious identity, sexual orientation, and behavioral patterns with legal or social risk. This is a content-sensitivity problem distinct from re-identification.
- **Non-English text.** The product targets English-language text only.
- **Adversarial robustness.** Deliberate obfuscation (leetspeak, character spacing, encoding tricks) is not addressed.
- **Long documents.** Text exceeding the model's context window is not handled natively; a sliding-window approach is recommended but not provided.
- **Compliance certification.** The product is not a complete privacy compliance solution and does not constitute legal guidance.

## 6. Constraints

- The model must be small enough to run on consumer hardware without specialized GPU.
- All training data must be synthetically generated to avoid handling real personal information.
- The project is a proof of concept; production hardening is not a goal.
- The Tier 2 model must be converted to GGUF format for use by the compiled binary.

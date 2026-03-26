# F-01: Quasi-Identifier Detection

**Priority:** Core

## What This Feature Does

The core capability of the product. Given free text, the system classifies which categories of quasi-identifier PII are present — information that is not directly identifying on its own but that, in combination with other details, can re-identify or locate a specific person.

For each analyzed text, the system returns:

- **Category labels** — which quasi-identifier categories are present in the text (location, workplace, routine, medical context, demographics, device identifier, credential, or quasi-identifier combination). No span-level extraction is performed.
- **Risk score** — an overall assessment of re-identification risk: LOW, MEDIUM, or HIGH.
- **Rationale** — a brief explanation of why the detected categories, especially in combination, create re-identification risk.

## Why It Exists

Existing PII detection tools handle direct identifiers (names, emails, SSNs) effectively but are fundamentally blind to contextual PII. A sentence like *"the only female cardiologist at St. Mary's in Tucson who does pediatric cases on Thursdays"* contains no canonical PII — yet it uniquely identifies a specific person. This feature fills that gap.

## Design Tradeoffs

- The system detects Tier 2 quasi-identifiers only. Tier 1 direct identifiers are delegated to existing tools; Tier 3 sensitive context is out of scope. This keeps the detection model focused on the unsolved problem rather than duplicating solved ones.
- The risk score is a simple three-level classification (LOW/MEDIUM/HIGH) rather than a numeric confidence score. This favors actionability — users need to know whether to act, not parse decimal probabilities.
- The rationale is brief by design. It explains quasi-identifier combinations in plain language to support user decision-making, not to provide exhaustive analysis.

## What This Feature Does Not Provide

- Detection of direct identifiers (names, emails, phone numbers, government IDs, etc.).
- Detection of sensitive context (medical diagnoses, legal history, political affiliation, etc.).
- Analysis of non-English text.
- Resistance to adversarial obfuscation.

## Acceptance Criteria

### AC-01: Category classification across all Tier 2 categories
**GIVEN** free text containing one or more quasi-identifiers from the taxonomy (LOCATION, WORKPLACE, ROUTINE, MEDICAL-CONTEXT, DEMOGRAPHIC, DEVICE-ID, CREDENTIAL, QUASI-ID)
**WHEN** the text is analyzed
**THEN** each quasi-identifier category present in the text is reported

### AC-02: Risk score assignment
**GIVEN** free text that has been analyzed for quasi-identifiers
**WHEN** the analysis completes
**THEN** an overall risk score of LOW, MEDIUM, or HIGH is returned

### AC-03: Rationale for quasi-identifier combinations
**GIVEN** free text containing two or more quasi-identifier categories whose combination increases re-identification risk
**WHEN** the text is analyzed
**THEN** a brief rationale explains how the combination of categories creates sensitivity

### AC-04: Clean text produces no findings
**GIVEN** free text containing no PII or quasi-identifiers
**WHEN** the text is analyzed
**THEN** no category labels are returned, the risk score is LOW, and no rationale is provided

### AC-05: Single quasi-identifier without combination
**GIVEN** free text containing a single quasi-identifier that does not uniquely identify a person on its own
**WHEN** the text is analyzed
**THEN** the category is reported, the risk score reflects the limited re-identification risk, and no combination rationale is needed

# F-10: Tier 1 Direct Identifier Detection

**Priority:** P1
**Requirements:** R-T1-01

## What This Feature Does

Detects direct identifiers — names, email addresses, phone numbers, government IDs, mailing addresses, financial account numbers, and dates of birth — in free text. These are the "obvious" PII that existing tools already handle well; this feature brings that capability into the same binary as the Tier 2 quasi-identifier detector so users get full-spectrum PII coverage from a single tool.

Detection produces span-level findings: each match includes the entity type, the matched text, and character offsets into the original input.

## Why It Exists

The product originally delegated Tier 1 detection to external tools like Presidio or spaCy. In practice, this meant users needed to install and configure a separate Python-based tool alongside the contextual PII tagger — defeating the goal of a lightweight, zero-dependency privacy gate.

By bundling a Rust-native NER library (redact-core), Tier 1 detection becomes part of the compiled binary with no additional setup. Users get both direct and contextual PII detection without managing two toolchains.

## Design Tradeoffs

- **Third-party library, not a custom model.** Tier 1 detection uses redact-core rather than a purpose-trained model. This keeps the project focused on its core differentiator (Tier 2 quasi-identifiers) while still covering the basics. The tradeoff is that entity type coverage and accuracy depend on redact-core's capabilities.
- **Separate hook, not merged output.** Tier 1 and Tier 2 run as independent Claude Code hooks with their own output formats. This avoids coupling the two detection tiers at the data model level and lets users enable or disable each tier independently.
- **No custom training or tuning.** The project does not train, fine-tune, or evaluate a Tier 1 model. Detection quality is inherited from redact-core as-is.

## What This Feature Does Not Provide

- Custom-trained Tier 1 models or fine-tuning.
- Tier 1 evaluation benchmarks or accuracy metrics managed by this project.
- Configurable entity type selection (all supported types are always active).
- Span-level output for Tier 2 quasi-identifiers (Tier 2 remains multilabel classification).

## Acceptance Criteria

### AC-01: Detects direct identifiers
**GIVEN** text containing one or more direct identifiers (name, email, phone, government ID, address, financial account number, or date of birth)
**WHEN** the Tier 1 scanner processes the text
**THEN** each direct identifier is reported as a finding with its entity type, matched text, and character offsets
*(Traces to R-T1-01)*

### AC-02: Clean text passes through
**GIVEN** text containing no direct identifiers
**WHEN** the Tier 1 scanner processes the text
**THEN** no findings are reported and the scan passes (exit 0)

### AC-03: Fully offline
**GIVEN** the Tier 1 scanner is running
**WHEN** detection is performed
**THEN** no data leaves the local machine
*(Traces to R-DET-04)*

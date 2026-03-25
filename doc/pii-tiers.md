# PII Tier Classification

This document defines the three-tier framework used throughout the project to classify PII by detection method and scope responsibility.

---

## Tier 1 — Direct Identifiers (Delegated)

Canonical PII that directly identifies an individual. Detection is delegated to existing tools (Microsoft Presidio, spaCy NER, regex patterns) and is **not part of the fine-tuned model's scope**.

| Label | Examples |
|-------|----------|
| `NAME` | Sarah Chen, Dr. Patel, my husband John |
| `EMAIL` | sarah@example.com |
| `PHONE` | (503) 555-0172 |
| `ADDRESS` | 412 Maple St, Apt 3B |
| `GOV-ID` | SSN 123-45-6789, passport A1234567 |
| `FINANCIAL` | Visa ending in 4242, routing 021000021 |
| `DOB` | born March 4 1985 |
| `BIOMETRIC` | fingerprint, face ID, retinal scan |

## Tier 2 — Quasi-Identifiers (QLoRA Model)

Information that is not directly identifying but can combine with other details to re-identify an individual. This is the **core focus of the fine-tuned model** and the project's primary differentiation from existing tools.

| Label | Examples | Notes |
|-------|----------|-------|
| `LOCATION` | downtown Portland, near the courthouse | Named and relative places |
| `WORKPLACE` | I work at Providence Hospital | Employers, offices, departments |
| `ROUTINE` | every Tuesday morning, after school pickup | Temporal behavioral patterns |
| `MEDICAL-CONTEXT` | my cardiologist, after my surgery last month | Health-adjacent without diagnosis details |
| `DEMOGRAPHIC` | the only woman in my department | Uniquely scoping demographic markers |
| `DEVICE-ID` | my iPhone 14 Pro, MAC address 00:1A:2B | Device and hardware identifiers |
| `CREDENTIAL` | my API key sk-..., password is... | Secrets, tokens, passwords |
| `QUASI-ID` | *(combination flag)* | Spans sensitive only in aggregate |

Tier 2 labels are formally specified in [specifications/entities.md](../specifications/entities.md), Section 1 (SpanLabel).

## Tier 3 — Sensitive Context (Out of Scope)

Information that is sensitive because its disclosure could cause harm — not because it identifies a person. Tier 3 is a content-sensitivity problem distinct from re-identification and is **out of scope for this project**.

| Label | Examples |
|-------|----------|
| `MEDICAL` | I have Type 2 diabetes, my HIV status |
| `LEGAL` | my DUI in 2019, the restraining order |
| `FINANCIAL-STATUS` | my bankruptcy filing, I owe $40k |
| `POLITICAL` | I voted for..., my party affiliation |
| `RELIGIOUS` | my church, I'm Muslim |
| `SEXUAL` | my partner, our relationship status |
| `BEHAVIORAL` | I was at the protest, I attend NA meetings |

Tier 3 labels are documented here for reference only and are not part of the model's training or inference.

---

## Scope Boundaries

| Tier | Responsibility | Rationale |
|------|---------------|-----------|
| 1 | Existing tools (Presidio, spaCy, regex) | Already solved; no need to duplicate |
| 2 | This project's fine-tuned model | Unsolved gap — requires contextual reasoning |
| 3 | Not addressed | Different problem class; would expand training scope significantly |

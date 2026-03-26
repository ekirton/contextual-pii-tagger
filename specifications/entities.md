# Entity Specifications

**Architecture:** [system-overview.md](../doc/architecture/system-overview.md), Section 3

---

## 1. SpanLabel

Enumeration of Tier 2 quasi-identifier categories.

### Values

`LOCATION` | `WORKPLACE` | `ROUTINE` | `MEDICAL-CONTEXT` | `DEMOGRAPHIC` | `DEVICE-ID` | `CREDENTIAL` | `QUASI-ID`

### Contract

- MAINTAINS: The set of valid values is fixed and matches the Tier 2 taxonomy.
- MAINTAINS: Values are uppercase strings with hyphens as separators.
- MAINTAINS: No value overlaps with Tier 1 labels (NAME, EMAIL, PHONE, ADDRESS, GOV-ID, FINANCIAL, DOB, BIOMETRIC).

---

## 2. RiskLevel

Enumeration of re-identification risk assessments.

### Values

`LOW` | `MEDIUM` | `HIGH`

### Contract

- MAINTAINS: Exactly three values, ordered by severity: LOW < MEDIUM < HIGH.
- MAINTAINS: Values are uppercase strings with no separators.

---

## 3. DetectionResult

The output of a single detection invocation. The model performs multilabel classification — it reports which quasi-identifier categories are present, not where they appear in the text.

### Fields

| Field | Type | Constraints |
|-------|------|-------------|
| `labels` | set\<SpanLabel\> | May be empty; each value must be a valid SpanLabel |
| `risk` | RiskLevel | Must be a valid RiskLevel value |
| `rationale` | string | May be empty |

### Contract

**REQUIRES:**
- `labels` is a set (possibly empty) of valid SpanLabel values.
- `risk` is a valid RiskLevel value.
- `rationale` is a string.

**ENSURES:**
- If `labels` is empty, `risk` is `LOW`.
- If `risk` is `LOW`, `rationale` is an empty string.
- If `risk` is `MEDIUM` or `HIGH`, `rationale` is a non-empty string.

**MAINTAINS:**
- The DetectionResult is immutable after construction.

---

## 4. Tier1Label

Enumeration of Tier 1 direct identifier categories detected by redact-ner.

### Values

`NAME` | `EMAIL` | `PHONE` | `ADDRESS` | `GOV-ID` | `FINANCIAL` | `DOB` | `BIOMETRIC`

### Contract

- MAINTAINS: Values are uppercase strings with hyphens as separators.
- MAINTAINS: No value overlaps with SpanLabel values (Section 1).
- MAINTAINS: The set of valid values matches the Tier 1 taxonomy defined in `doc/pii-tiers.md`.

---

## 5. Tier1Finding

A single span-level finding from Tier 1 detection.

### Fields

| Field | Type | Constraints |
|-------|------|-------------|
| `label` | Tier1Label | Must be a valid Tier1Label value |
| `text` | string | Non-empty; the matched text from the input |
| `start` | integer | >= 0; character offset of the match start |
| `end` | integer | > start; character offset of the match end (exclusive) |

### Contract

**REQUIRES:**
- `label` is a valid Tier1Label value.
- `text` is a non-empty string.
- `start` >= 0.
- `end` > `start`.

**ENSURES:**
- `end - start` equals the length of `text` in characters.

**MAINTAINS:**
- The Tier1Finding is immutable after construction.

---

## 6. Example

A single record in the dataset.

### Fields

| Field | Type | Constraints |
|-------|------|-------------|
| `id` | string | Non-empty, unique within the dataset |
| `text` | string | Non-empty |
| `labels` | set\<SpanLabel\> | May be empty; each value must be a valid SpanLabel |
| `risk` | RiskLevel | Valid RiskLevel |
| `rationale` | string | May be empty |
| `is_hard_negative` | boolean | |
| `split` | string | One of: `train`, `validation`, `test` |
| `domain` | string | One of: `medical`, `scheduling`, `workplace`, `personal` |
| `source` | string | One of: `template`, `llm-augmented`, `hard-negative` |

### Contract

**REQUIRES:**
- `id` is non-empty and unique within the dataset.
- `text` is non-empty.
- `split` is one of the three valid values.
- `domain` is one of the four valid values.
- `source` is one of the three valid values.

**ENSURES:**
- If `is_hard_negative` is true: `labels` is empty, `risk` is `LOW`, `rationale` is an empty string, and `source` is `hard-negative`.
- The DetectionResult invariants (Section 3) hold for the `labels`, `risk`, and `rationale` fields.

**MAINTAINS:**
- `id` follows the pattern `{split}-{zero-padded-number}` (e.g., `train-00001`, `test-04999`).

---

## 7. EvaluationReport

Output of the evaluation pipeline.

### Fields

| Field | Type | Constraints |
|-------|------|-------------|
| `model_name` | string | Non-empty |
| `test_set_size` | integer | > 0 |
| `multilabel_f1` | float | 0.0 <= value <= 1.0 |
| `f1_by_label` | map\<SpanLabel, float\> | One entry per SpanLabel; each value 0.0 <= v <= 1.0 |
| `risk_accuracy` | float | 0.0 <= value <= 1.0 |
| `false_negative_rate` | float | 0.0 <= value <= 1.0 |
| `quasi_id_f1` | float | 0.0 <= value <= 1.0 |
| `hard_negative_precision` | float | 0.0 <= value <= 1.0 |

### Contract

**REQUIRES:**
- `model_name` is non-empty.
- `test_set_size` > 0.

**ENSURES:**
- `f1_by_label` contains exactly one entry for each SpanLabel value (8 entries).
- `multilabel_f1` equals the macro-average of all values in `f1_by_label`.
- All float fields are in the range [0.0, 1.0].

**MAINTAINS:**
- The EvaluationReport is immutable after construction.

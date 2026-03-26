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

## 4. Example

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

## 5. EvaluationReport

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

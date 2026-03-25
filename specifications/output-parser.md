# Output Parser Specification

**Architecture:** [inference-pipeline.md](../doc/architecture/inference-pipeline.md), Section 2.3
**Features:** F-01, F-04

---

## 1. parse_output

```
parse_output(raw_output: string) -> DetectionResult
```

Transforms the model's raw completion string into a validated DetectionResult.

**REQUIRES:**
- `raw_output` is a string (may be empty, malformed, or valid JSON).

**ENSURES:**
- Always returns a valid DetectionResult (never raises on malformed input).
- The returned DetectionResult satisfies all invariants from entities.md Section 3.
- All labels in the result are valid SpanLabel values.

**MAINTAINS:**
- The parser is stateless and deterministic.

---

## 2. Parsing Stages

The parser applies the following stages in order. Each stage either succeeds and passes to the next, or triggers a fallback.

### 2.1 JSON Extraction

**REQUIRES:** `raw_output` is a string.

**ENSURES:**
- If `raw_output` contains valid JSON, it is parsed into a dictionary.
- If `raw_output` contains JSON embedded in surrounding text (e.g., markdown code fences), the JSON is extracted first.
- If JSON is malformed, repair is attempted (Stage 2.2).
- If extraction and repair both fail, the fallback DetectionResult is returned (Stage 2.5).

### 2.2 JSON Repair

**REQUIRES:** A string that failed strict JSON parsing.

**ENSURES:**
- Attempts to fix common malformations:
  - Unclosed brackets and braces.
  - Trailing commas.
  - Single quotes instead of double quotes.
  - Unquoted keys.
- If repair produces valid JSON, proceeds to Stage 2.3.
- If repair fails, the fallback DetectionResult is returned (Stage 2.5).

### 2.3 Field Extraction

**REQUIRES:** A valid JSON dictionary.

**ENSURES:**
- Extracts `labels` (list), `risk` (string), and `rationale` (string).
- Missing `labels` key → empty set.
- Missing `risk` key → `LOW`.
- Missing `rationale` key → empty string.
- `risk` value not in RiskLevel enum → `LOW`.

### 2.4 Label Validation

**REQUIRES:** A list of raw label strings.

**ENSURES:**
- Each label string is checked against the SpanLabel enum.
- Labels not in the SpanLabel enum are **dropped** with a warning.
- Duplicate labels are collapsed (the result is a set).

### 2.5 Fallback

**REQUIRES:** All prior stages have failed or produced no usable output.

**ENSURES:**
- Returns `DetectionResult(labels=set(), risk=LOW, rationale="")`.

---

## 3. Consistency Enforcement

After all stages complete, the parser enforces DetectionResult invariants:

**ENSURES:**
- If `labels` is empty → `risk` is set to `LOW`, `rationale` is set to `""`.
- If `risk` is `LOW` → `rationale` is set to `""`.
- If `risk` is `MEDIUM` or `HIGH` and `labels` has >= 2 entries and `rationale` is empty → `rationale` is set to `"Multiple quasi-identifiers detected."` (generic fallback).

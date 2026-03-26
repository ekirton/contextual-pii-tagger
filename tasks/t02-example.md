# T-02: Example and EvaluationReport Entities

**File:** `src/contextual_pii_tagger/example.py`
**Spec:** [entities.md](../specifications/entities.md) §4-5
**Depends on:** T-01 (entities.py)

## Scope

Implement the dataset and evaluation entities: `Example`, `EvaluationReport`.

## Deliverables

1. **Example dataclass** — fields: `id`, `text`, `labels`, `risk`, `rationale`, `is_hard_negative`, `split`, `domain`, `source`. Validate on construction:
   - `id` non-empty, matches pattern `{split}-{zero-padded-number}`
   - `text` non-empty
   - `split` in {train, validation, test}
   - `domain` in {medical, scheduling, workplace, personal}
   - `source` in {template, llm-augmented, hard-negative}
   - Hard negative invariant: `is_hard_negative` → empty labels, LOW risk, empty rationale, source `hard-negative`
   - DetectionResult invariants hold for labels/risk/rationale

2. **EvaluationReport dataclass** (frozen) — all fields from spec. Validate:
   - `f1_by_label` has exactly 8 entries (one per SpanLabel)
   - `multilabel_f1` equals macro-average of per-label values
   - All floats in [0.0, 1.0]

3. **Serialization** — `to_dict()` / `from_dict()` for JSON Lines I/O. Labels serialized as a sorted list.

## Acceptance Criteria

- All REQUIRES/ENSURES from entities.md §4-5 are enforced.
- EvaluationReport verifies macro-average consistency.

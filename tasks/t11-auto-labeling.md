# T-11: Auto-Labeling Validation

**File:** `src/contextual_pii_tagger/data/validate_labels.py`
**Spec:** [data-generation.md](../specifications/data-generation.md) §4
**Depends on:** T-01 (entities.py), T-02 (example.py)

## Scope

Implement `validate_labels(examples) -> list[Example]`.

## Deliverables

1. **Second-pass LLM validation** — For each Example, call a frontier LLM to verify category labels, risk, and rationale.

2. **Label verification** — Check that each label is correct for the text content. Identify missing labels (categories present but not annotated).

3. **RiskLevel validation** — Ensure risk matches the combination of labels.

4. **Rationale generation** — Generate rationale for MEDIUM/HIGH risk examples missing one.

5. **Disagreement handling** — Remove examples where validation cannot resolve disagreements (caller must regenerate to meet count targets).

## Acceptance Criteria

- All returned Examples satisfy full Example invariants.
- Removed examples logged with count.

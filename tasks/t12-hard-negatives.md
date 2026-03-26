# T-12: Hard Negative Injection

**File:** `src/contextual_pii_tagger/data/hard_negatives.py`
**Spec:** [data-generation.md](../specifications/data-generation.md) §5
**Depends on:** T-01 (entities.py), T-02 (example.py)

## Scope

Implement `inject_hard_negatives(examples, ratio) -> list[Example]`.

## Deliverables

1. **Hard negative generation** — Generate texts containing location, time, organization, or person references that are not PII in context. Categories: historical references, fictional characters, public figures, generic statements, hypotheticals.

2. **Ratio enforcement** — Compute count needed so hard negatives = 10% of total dataset. Distribute proportionally across splits.

3. **Example construction** — Each hard negative: `is_hard_negative: true`, `labels: frozenset()`, `risk: LOW`, `rationale: ""`, `source: hard-negative`.

## Acceptance Criteria

- Hard negatives are exactly 10% of each split.
- All hard negative Examples satisfy Example invariants.
- Hard negative texts contain PII-like tokens but are genuinely not PII.

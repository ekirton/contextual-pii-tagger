# T-01: Core Entities

**File:** `src/contextual_pii_tagger/entities.py`
**Spec:** [entities.md](../specifications/entities.md) §1-3
**Depends on:** Nothing

## Scope

Implement the three core runtime entities: `SpanLabel`, `RiskLevel`, `DetectionResult`.

## Deliverables

1. **SpanLabel enum** — 8 values matching Tier 2 taxonomy. String enum with uppercase hyphenated values.

2. **RiskLevel enum** — 3 values: LOW, MEDIUM, HIGH. String enum, ordered by severity.

3. **DetectionResult dataclass** (frozen/immutable) — fields: `labels` (frozenset of SpanLabel), `risk` (RiskLevel), `rationale` (str). Enforce invariants on construction:
   - Empty labels → risk LOW, rationale empty
   - Risk LOW → rationale empty
   - Risk MEDIUM/HIGH with 2+ labels → rationale non-empty

4. **Serialization** — `DetectionResult.to_dict()` and `DetectionResult.from_dict()` for JSON round-tripping. Labels serialized as a sorted list for determinism.

## Acceptance Criteria

- All REQUIRES/ENSURES from entities.md §1-3 are enforced.
- `ValueError` raised on invalid construction arguments.
- Entities are immutable after construction.
- JSON serialization round-trips without data loss.

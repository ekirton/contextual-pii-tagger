# T-03: Output Parser

**File:** `src/contextual_pii_tagger/output_parser.py`
**Spec:** [entities.md](../specifications/entities.md) §3
**Depends on:** T-01 (entities.py)

## Scope

Implement `parse_output(raw_output) -> DetectionResult` with all five parsing stages.

## Deliverables

1. **JSON extraction** — Extract JSON from raw model output. Handle markdown code fences and surrounding text.

2. **JSON repair** — Fix common malformations: unclosed brackets, trailing commas, single quotes, unquoted keys.

3. **Field extraction** — Extract `labels`, `risk`, `rationale` with defaults for missing keys.

4. **Label validation** — For each label string:
   - Drop if not a valid SpanLabel
   - Collapse duplicates to a set

5. **Consistency enforcement** — Enforce DetectionResult invariants (empty labels → LOW, LOW → empty rationale, generic fallback rationale).

6. **Fallback** — Return `DetectionResult(labels=frozenset(), risk=LOW, rationale="")` when all parsing fails.

## Acceptance Criteria

- Never raises on any input (total function).
- Always returns a valid DetectionResult.
- Invalid labels dropped with warning.
- Duplicate labels collapsed.
- Malformed JSON cases handled (per spec §2.2).

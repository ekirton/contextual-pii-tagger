# T-09: Template-Based Generation

**File:** `src/contextual_pii_tagger/data/templates.py`
**Spec:** [data-generation.md](../specifications/data-generation.md) §2
**Depends on:** T-01 (entities.py), T-02 (example.py)

## Scope

Implement `generate_from_templates(templates_dir, count) -> list[Example]`.

## Deliverables

1. **YAML template loader** — Parse template files with domain, patterns, labels, risk, and rationale_template fields. Validate SpanLabel values.

2. **Slot filling** — Use Faker to fill template slots with synthetic values.

3. **Label set construction** — Each template pattern specifies which SpanLabel categories are present. Build the Example's `labels` frozenset from the template annotation.

4. **Domain distribution** — Distribute examples across available templates/domains, not all from one file.

5. **Example construction** — Build valid Example records with `source: template`, correct label sets.

## Acceptance Criteria

- All Example invariants hold.
- No real personal data (Faker only).
- All four domains represented when templates exist for each.

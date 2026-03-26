# T-23: Domain Template Files

**Files:** `src/contextual_pii_tagger/data/templates/*.yaml`
**Spec:** [data-generation.md](../specifications/data-generation.md) §2.2
**Depends on:** T-01 (entities.py — for valid SpanLabel values)

## Scope

Create YAML template files for each of the four domains.

## Deliverables

1. **medical.yaml** — Templates for doctor visits, specialist referrals, health-adjacent questions. Labels: MEDICAL-CONTEXT, WORKPLACE, ROUTINE.

2. **scheduling.yaml** — Templates for commute patterns, recurring appointments, school schedules. Labels: ROUTINE, LOCATION.

3. **workplace.yaml** — Templates for office locations, departments, job roles, team descriptions. Labels: WORKPLACE, DEMOGRAPHIC, LOCATION.

4. **personal.yaml** — Templates for introductions, neighborhood descriptions, family context. Labels: LOCATION, ROUTINE, DEMOGRAPHIC.

Each template file must:
- Have a `domain` field matching its domain
- Include multiple patterns (10+ per file for diversity)
- Each pattern has `text` with `{SLOT}` placeholders, `spans` with label annotations, `risk`, and `rationale_template`
- All span labels are valid SpanLabels
- Cover multiple SpanLabel combinations per domain

## Acceptance Criteria

- All four domain files exist and parse as valid YAML.
- Every span label is a valid SpanLabel.
- At least 10 patterns per domain.
- All 8 SpanLabel values appear across the four files collectively.

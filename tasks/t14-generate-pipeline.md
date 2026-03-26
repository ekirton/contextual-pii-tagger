# T-14: Generation Pipeline Orchestrator

**File:** `src/contextual_pii_tagger/data/generate.py`
**Spec:** [data-generation.md](../specifications/data-generation.md) §1
**Depends on:** T-09, T-10, T-11, T-12, T-13

## Scope

Implement `generate_dataset(config) -> Dataset` orchestrating all five stages.

## Deliverables

1. **Pipeline orchestration** — Call stages 1-5 in sequence:
   - Stage 1: `generate_from_templates`
   - Stage 2: `generate_from_llm`
   - Stage 3: `validate_labels`
   - Stage 4: `inject_hard_negatives`
   - Stage 5: `select_review_sample` (output sample IDs for human review)

2. **Split assignment** — Shuffle and assign examples to train/validation/test (80:10:10) before human review.

3. **ID assignment** — Assign IDs in `{split}-{zero-padded-number}` format.

4. **Count verification** — Verify final dataset: 12,500 total, correct split sizes, 10% hard negatives per split.

5. **CLI entry point** — Accept config path, run pipeline, write output.

## Acceptance Criteria

- Output dataset satisfies all generate_dataset ENSURES from spec.
- All four domains and all eight SpanLabels represented.
- No real personal data.

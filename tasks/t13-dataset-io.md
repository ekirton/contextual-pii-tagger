# T-13: Dataset I/O

**File:** `src/contextual_pii_tagger/data/dataset_io.py`
**Spec:** [data-generation.md](../specifications/data-generation.md) §7
**Depends on:** T-01 (entities.py), T-02 (example.py)

## Scope

Implement `write_dataset(dataset, output_dir)` and `read_dataset(input_dir) -> list[Example]`.

## Deliverables

1. **write_dataset** — Write three JSONL files: `train.jsonl`, `validation.jsonl`, `test.jsonl`. One Example per line, compact JSON.

2. **read_dataset** — Read JSONL files back into Example records. Validate each record on load.

3. **Split validation** — Verify expected split proportions (approximately 80:10:10) on write.

## Acceptance Criteria

- Round-trip: write then read produces identical Example records.
- Each line is valid JSON deserializable to a valid Example.
- File counts match spec exactly.

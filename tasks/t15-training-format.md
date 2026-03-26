# T-15: Training Data Formatting

**File:** `src/contextual_pii_tagger/train/data_utils.py`
**Spec:** [training.md](../specifications/training.md) §1-2
**Depends on:** T-01 (entities.py), T-02 (example.py), T-05 (prompt.py)

## Scope

Implement `format_example` and `prepare_dataset` for the training loop.

## Deliverables

1. **format_example(example, tokenizer)** — Convert Example to prompt-completion pair:
   - Prompt uses shared PROMPT_TEMPLATE from prompt.py
   - Completion is compact JSON of labels/risk/rationale
   - Skip (don't truncate) examples exceeding 1,024 tokens
   - Log skipped example count

2. **prepare_dataset(dataset_path, tokenizer)** — Load train.jsonl, format all examples, shuffle, return formatted dataset compatible with SFTTrainer.

## Acceptance Criteria

- Prompt template identical to inference template (single source in prompt.py).
- Hard negatives formatted with empty labels, LOW risk, empty rationale.
- Skipped examples counted and logged.
- Output compatible with `trl.SFTTrainer`.

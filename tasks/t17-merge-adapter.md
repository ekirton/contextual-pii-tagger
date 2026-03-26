# T-17: Merge Adapter

**File:** `src/contextual_pii_tagger/train/merge.py`
**Spec:** [training.md](../specifications/training.md) §4
**Depends on:** T-16 (train.py)

## Scope

Implement `merge_adapter(base_model_path, adapter_path, output_path)`.

## Deliverables

1. **Merge logic** — Load base model, apply adapter, merge weights, save as standalone model.

2. **Tokenizer copy** — Include tokenizer files in output directory.

3. **Verification** — After merge, verify that loading the merged model (without peft) produces the same output as base+adapter for a sample input.

4. **CLI entry point** — Accept base path, adapter path, output path.

## Acceptance Criteria

- Merged model loadable without peft library.
- Inference results identical to base+adapter.
- Tokenizer included in output.

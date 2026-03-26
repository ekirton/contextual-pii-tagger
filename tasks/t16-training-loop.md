# T-16: Training Loop

**File:** `src/contextual_pii_tagger/train/train.py`
**Spec:** [training.md](../specifications/training.md) §3, §5
**Depends on:** T-15 (data_utils.py), T-24 (config.yaml)

## Scope

Implement `train(config, dataset, base_model_path) -> TrainingOutput` and `validate_epoch`.

## Deliverables

1. **Model loading** — Load Phi-3 Mini with 4-bit NF4 quantization via bitsandbytes.

2. **LoRA setup** — Apply LoRA adapter: r=16, alpha=32, dropout=0.05, target modules [q_proj, v_proj, k_proj, o_proj].

3. **Training loop** — Use SFTTrainer with: 3 epochs, batch size 4, gradient accumulation 4, lr 2e-4, cosine scheduler, warmup 0.05, max_seq_length 1024, bf16.

4. **validate_epoch** — Run validation at end of each epoch, report loss, category_f1, risk_accuracy.

5. **Output** — Save adapter weights, training logs.

6. **CLI entry point** — Accept config.yaml path, run training.

## Acceptance Criteria

- Base model weights not modified.
- Adapter weights saved and loadable.
- Validation metrics computed per evaluation spec.
- bf16 precision used.

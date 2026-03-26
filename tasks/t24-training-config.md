# T-24: Training Configuration File

**File:** `src/contextual_pii_tagger/train/config.yaml`
**Spec:** [training.md](../specifications/training.md) §3
**Depends on:** Nothing

## Scope

Create the training configuration YAML matching the spec hyperparameters.

## Deliverables

1. **config.yaml** with all values from training-pipeline.md §4-5:
   ```yaml
   base_model: microsoft/Phi-3-mini-4k-instruct
   load_in_4bit: true
   bnb_4bit_compute_dtype: bfloat16
   bnb_4bit_quant_type: nf4
   lora_r: 16
   lora_alpha: 32
   lora_dropout: 0.05
   lora_target_modules: [q_proj, v_proj, k_proj, o_proj]
   num_train_epochs: 3
   per_device_train_batch_size: 4
   gradient_accumulation_steps: 4
   learning_rate: 2e-4
   lr_scheduler_type: cosine
   warmup_ratio: 0.05
   max_seq_length: 1024
   output_dir: ./output/contextual-pii-tagger
   ```

2. **Config loader** — Function in train.py to load and validate config values.

## Acceptance Criteria

- All values match spec exactly.
- Config loads without error.

# Training Specification

**Architecture:** [training-pipeline.md](../doc/architecture/training-pipeline.md)

---

## 1. format_example

```
format_example(example: Example, tokenizer: Tokenizer) -> FormattedExample
```

Converts an Example record into a tokenized prompt-completion pair for training.

**REQUIRES:**
- `example` is a valid Example record (satisfies all Example invariants).
- `tokenizer` is a loaded Phi-3 tokenizer.

**ENSURES:**
- The prompt portion follows the template:
  ```
  <|user|>
  Classify which quasi-identifier PII categories are present in the
  following text. Return the list of category labels from the taxonomy,
  an overall risk score (LOW/MEDIUM/HIGH), and a brief rationale.

  Text: {example.text}
  <|end|>
  ```
- The completion portion is:
  ```
  <|assistant|>
  {"labels": [...], "risk": "...", "rationale": "..."}
  <|end|>
  ```
- The JSON in the completion is compact (no extra whitespace).
- For hard negatives: `labels` is `[]`, `risk` is `"LOW"`, `rationale` is `""`.
- The combined token count (prompt + completion) does not exceed 1,024 tokens.
- If the combined tokens exceed 1,024, the example is **skipped** (not truncated), and a warning is logged.

**MAINTAINS:**
- The prompt template is identical across all examples.

---

## 2. prepare_dataset

```
prepare_dataset(dataset_path: string, tokenizer: Tokenizer) -> FormattedDataset
```

Loads and formats the training split for the training loop.

**REQUIRES:**
- `dataset_path` points to a directory containing `train.jsonl`.
- `tokenizer` is a loaded Phi-3 tokenizer.

**ENSURES:**
- Each Example in `train.jsonl` is formatted via `format_example`.
- Examples exceeding 1,024 tokens are excluded (with count logged).
- The returned FormattedDataset is shuffled.

---

## 3. train

```
train(config: TrainingConfig, dataset: FormattedDataset, base_model_path: string) -> TrainingOutput
```

Runs the QLoRA fine-tuning loop.

**REQUIRES:**
- `config` contains all hyperparameters as specified in training-pipeline.md Sections 4-5:
  - quantization: 4-bit NF4
  - lora_r: 16, lora_alpha: 32, lora_dropout: 0.05
  - target_modules: [q_proj, v_proj, k_proj, o_proj]
  - epochs: 3, batch_size: 4, gradient_accumulation: 4
  - learning_rate: 2e-4, scheduler: cosine, warmup_ratio: 0.05
  - max_seq_length: 1024
- `dataset` is a formatted training dataset.
- `base_model_path` points to `microsoft/Phi-3-mini-4k-instruct` weights.

**ENSURES:**
- Returns a TrainingOutput containing:
  - LoRA adapter weights (saved to `config.output_dir`).
  - Training logs (loss per step, learning rate per step).
  - Per-epoch validation metrics (if validation split path is provided in config).
- The adapter weights are compatible with the base model for later merging.
- Training precision is bf16.

**MAINTAINS:**
- The base model weights are not modified (QLoRA trains only the adapter parameters).
- The training loop uses SFTTrainer from the `trl` library.

---

## 4. merge_adapter

```
merge_adapter(base_model_path: string, adapter_path: string, output_path: string) -> void
```

Merges the LoRA adapter into the base model for standalone inference.

**REQUIRES:**
- `base_model_path` points to the Phi-3 Mini base model.
- `adapter_path` points to the trained LoRA adapter weights.
- `output_path` is a writable directory.

**ENSURES:**
- Writes a merged model to `output_path` that produces identical inference results to the base+adapter combination.
- The merged model can be loaded without the `peft` library.
- The merged model includes the tokenizer files.

---

## 5. validate_epoch

```
validate_epoch(model: Model, validation_dataset: FormattedDataset) -> EpochMetrics
```

Runs validation at the end of each training epoch.

**REQUIRES:**
- `model` is the current state of the model (base + adapter after N epochs).
- `validation_dataset` is the formatted validation split.

**ENSURES:**
- Returns EpochMetrics containing:
  - `validation_loss`: float (average loss over the validation set).
  - `multilabel_f1`: float (aggregate multilabel F1 across all SpanLabels).
  - `risk_accuracy`: float (fraction of correct RiskLevel predictions).
- Metrics are computed using the same logic as the evaluation pipeline (evaluation.md).

**MAINTAINS:**
- Validation does not modify the model weights.

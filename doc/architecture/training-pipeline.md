# Training Pipeline

**Features:** F-01 (Quasi-Identifier Detection), F-04 (Detection Interface)
**Requirements:** R-DET-01, R-DET-02, R-DET-03

---

## 1. Pipeline Overview

The training pipeline takes the dataset (train split) and the base model, applies QLoRA fine-tuning, and produces a LoRA adapter that transforms the base model into a quasi-identifier tagger.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Dataset    │───>│  Formatting  │───>│   QLoRA      │───>│   Adapter    │
│ (train split)│    │  & Tokenize  │    │  Fine-Tuning │    │   Weights    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                              ^
                                              │
                                     ┌────────────────┐
                                     │   Base Model   │
                                     │  (Phi-3 Mini)  │
                                     └────────────────┘
```

## 2. Base Model

| Property | Value |
|----------|-------|
| Model | Microsoft Phi-3 Mini |
| Variant | `microsoft/Phi-3-mini-4k-instruct` |
| Parameters | 3.8B |
| License | MIT |
| Context window | 4,096 tokens |

The instruction-tuned variant is selected because the tagging task uses an instruction-following prompt format. The base model's existing instruction-following capability provides a strong starting point.

## 3. Example Formatting

Each Example record is formatted into the model's chat template as a prompt-completion pair.

**Prompt (user turn):**

```
<|user|>
Classify which quasi-identifier PII categories are present in the
following text. Return the list of category labels from the taxonomy,
an overall risk score (LOW/MEDIUM/HIGH), and a brief rationale.

Text: {example.text}
<|end|>
```

**Completion (assistant turn):**

```
<|assistant|>
{
  "labels": ["{label}", ...],
  "risk": "{example.risk}",
  "rationale": "{example.rationale}"
}
<|end|>
```

For hard negatives (`is_hard_negative: true`), the completion contains an empty label list, risk `LOW`, and an empty rationale.

## 4. QLoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Quantization | 4-bit NF4 | Reduces base model VRAM from ~8GB to ~3GB |
| LoRA rank (r) | 16 | Sufficient capacity for the classification task |
| LoRA alpha | 32 | Standard 2x rank scaling |
| Target modules | q_proj, v_proj, k_proj, o_proj | All attention projection layers |
| LoRA dropout | 0.05 | Light regularization to prevent overfitting |
| Trainable parameters | ~8M (~0.2% of base) | Highly parameter-efficient |
| Training precision | bf16 | Numeric stability on modern GPUs |

## 5. Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 3 | Sufficient for behavioral adaptation without overfitting |
| Per-device batch size | 4 | Fits in 16GB VRAM with QLoRA |
| Gradient accumulation steps | 4 | Effective batch size of 16 |
| Learning rate | 2e-4 | Standard for QLoRA fine-tuning |
| LR scheduler | Cosine | Smooth decay to zero |
| Warmup ratio | 0.05 | Brief warmup to stabilize early training |
| Max sequence length | 1,024 tokens | Covers the vast majority of prompt lengths |

## 6. Training Stack

| Library | Role |
|---------|------|
| `transformers` | Model loading, tokenization |
| `peft` | LoRA/QLoRA adapter creation and management |
| `trl` (SFTTrainer) | Supervised fine-tuning training loop |
| `bitsandbytes` | 4-bit NF4 quantization |
| `datasets` | Dataset loading and preprocessing |
| `accelerate` | Mixed-precision and multi-GPU support |

## 7. Outputs

| Artifact | Description |
|----------|-------------|
| LoRA adapter weights | The trained adapter (~32MB), saved separately from the base model |
| Training logs | Loss curves, learning rate schedule, and per-epoch validation metrics |
| Merged model weights | Base model with adapter merged, ready for standalone inference without `peft` |

The merged variant is produced after training as an export step. It enables inference without the `peft` library, which simplifies the Detection Interface dependency tree.

## 8. Validation During Training

At the end of each epoch, the training pipeline evaluates on the validation split (5,000 examples) and reports:
- Validation loss
- Multilabel F1 (aggregate)
- Risk score accuracy

These metrics are used for early stopping or hyperparameter decisions, not as the final evaluation (which uses the held-out test split via the Evaluation Pipeline).

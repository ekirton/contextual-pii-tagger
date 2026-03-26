# System Overview

**Features:** F-01, F-02, F-03, F-06

---

## 1. System Boundary Diagram

The system consists of four pipelines that execute sequentially during development.

```
┌─────────────────────────────────────────────────────────────────┐
│                     DEVELOPMENT PIPELINES                       │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   Data Gen   │───>│   Training   │───>│    Evaluation    │  │
│  │   Pipeline   │    │   Pipeline   │    │    Pipeline      │  │
│  └──────┬───────┘    └──────────────┘    └──────────────────┘  │
│         │                                                       │
│         v                                                       │
│  ┌──────────────┐                                              │
│  │ Human Review │                                              │
│  │  Workflow    │                                              │
│  └──────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Component Boundaries

| Component | Responsibility | Inputs | Outputs |
|-----------|---------------|--------|---------|
| Data Generation Pipeline | Produce labeled Example records from synthetic sources | Domain templates, Faker library, frontier LLM API | Dataset file (train/validation/test splits) |
| Human Review Workflow | Spot-check 1% of Example records for label correctness | Dataset file, annotation interface | Corrected Dataset file |
| Training Pipeline | Fine-tune the base model using QLoRA on the training split | Dataset file (train split), base model weights | LoRA adapter weights |
| Evaluation Pipeline | Measure model and XGBoost baseline against success metrics | Dataset file (test split), adapter weights, XGBoost model | EvaluationReport |

## 3. Entity Definitions

These entities are the canonical data shapes used across all pipelines. All downstream documents must use these exact names, fields, and types.

### 3.1 SpanLabel (enum)

Tier 2 quasi-identifier categories detected by the model.

| Value | Description |
|-------|-------------|
| `LOCATION` | Named and relative places |
| `WORKPLACE` | Employers, offices, departments |
| `ROUTINE` | Temporal behavioral patterns |
| `MEDICAL-CONTEXT` | Health-adjacent without diagnosis details |
| `DEMOGRAPHIC` | Uniquely scoping demographic markers |
| `DEVICE-ID` | Device and hardware identifiers |
| `CREDENTIAL` | Secrets, tokens, passwords |
| `QUASI-ID` | Combination flag — spans sensitive only in aggregate |

### 3.2 RiskLevel (enum)

| Value | Description |
|-------|-------------|
| `LOW` | No meaningful re-identification risk |
| `MEDIUM` | Some quasi-identifiers present; limited re-identification risk in isolation |
| `HIGH` | Quasi-identifier combination sufficient to re-identify or locate an individual |

### 3.3 DetectionResult

The output of a single detection invocation. The model performs multilabel classification — it reports which quasi-identifier categories are present, not where they appear in the text.

| Field | Type | Description |
|-------|------|-------------|
| `labels` | set\<SpanLabel\> | Quasi-identifier categories present in the text (empty set if none) |
| `risk` | RiskLevel | Overall re-identification risk assessment |
| `rationale` | string | Brief explanation of quasi-identifier combinations (empty string if risk is LOW) |

### 3.4 Example

A single record in the dataset (training, validation, or test).

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for the example |
| `text` | string | The input free text |
| `labels` | set\<SpanLabel\> | Ground-truth quasi-identifier categories present |
| `risk` | RiskLevel | Ground-truth risk score |
| `rationale` | string | Ground-truth rationale |
| `is_hard_negative` | boolean | True if this example is a hard negative (no PII despite appearances) |
| `split` | string | One of: `train`, `validation`, `test` |
| `domain` | string | Generation domain: `medical`, `scheduling`, `workplace`, `personal` |
| `source` | string | Generation method: `template`, `llm-augmented`, `hard-negative` |

### 3.5 EvaluationReport

Output of the evaluation pipeline.

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | string | Identifier for the evaluated model |
| `test_set_size` | integer | Number of Example records evaluated |
| `multilabel_f1` | float | Aggregate multilabel F1 across all SpanLabel categories |
| `f1_by_label` | map\<SpanLabel, float\> | F1 broken down per SpanLabel |
| `risk_accuracy` | float | Fraction of correct RiskLevel predictions |
| `false_negative_rate` | float | Fraction of texts with PII classified as clean (empty label set) |
| `quasi_id_f1` | float | F1 on QUASI-ID label only |
| `hard_negative_precision` | float | Fraction of hard negatives correctly classified as clean |

## 4. External Dependencies

| Dependency | Used By | Purpose |
|------------|---------|---------|
| Microsoft Phi-3 Mini (3.8B) | Training Pipeline | Base model for QLoRA fine-tuning |
| Faker library | Data Generation Pipeline | Synthetic data for template-based generation |
| Frontier LLM API (Claude / GPT-4) | Data Generation Pipeline | LLM-augmented example generation and auto-labeling |
| Label Studio | Human Review Workflow | Annotation interface for spot-checking |

## 5. Key Constraints

- **Consumer hardware.** Training requires a single GPU with 16GB VRAM. Evaluation runs on CPU.
- **No real PII.** No pipeline ingests, stores, or processes real personal data at any stage.
- **1,024 token limit.** Input text exceeding the model's context window (1,024 tokens) is truncated. The system does not implement sliding windows.

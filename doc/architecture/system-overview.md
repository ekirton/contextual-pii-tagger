# System Overview

**Features:** F-01, F-02, F-03, F-04, F-06, F-09, F-10, F-11
**Requirements:** All

---

## 1. System Boundary Diagram

The system consists of four pipelines that execute sequentially during development, plus two runtime components that serve end users.

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

┌─────────────────────────────────────────────────────────────────┐
│                     RUNTIME COMPONENTS                          │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────┐  │
│  │  Detection   │    │  Claude Code Hooks                   │  │
│  │  Interface   │    │  (UserPromptSubmit, PreToolUse,      │  │
│  │  (Python)    │    │   PostToolUse)                       │  │
│  └──────────────┘    └──────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  PII Scanner Binary (Rust)                               │  │
│  │  Tier 1: redact-ner  |  Tier 2: llama.cpp + GGUF        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Component Boundaries

| Component | Responsibility | Inputs | Outputs |
|-----------|---------------|--------|---------|
| Data Generation Pipeline | Produce 50,000 labeled Example records from synthetic sources | Domain templates, Faker library, frontier LLM API | Dataset file (train/validation/test splits) |
| Human Review Workflow | Spot-check 1% of Example records for label correctness | Dataset file, annotation interface | Corrected Dataset file |
| Training Pipeline | Fine-tune the base model using QLoRA on the training split | Dataset file (train split), base model weights | LoRA adapter weights |
| Evaluation Pipeline | Measure model and XGBoost baseline against success metrics | Dataset file (test split), adapter weights, XGBoost model | EvaluationReport |
| Detection Interface | Accept text, run inference, return structured DetectionResult | Input text string, merged model weights | DetectionResult |
| Claude Code Hooks | Intercept Claude Code data flows, invoke Detection Interface, block on PII | Hook event payload (text content) | Exit code 0 (pass) or exit code 2 (block) + stderr findings |
| PII Scanner Binary | Compiled Rust binary combining Tier 1 (redact-ner) and Tier 2 (llama.cpp) detection, replacing the Python inference and hook layer for runtime use | Hook event payload (text content), GGUF model file | Exit code 0 (pass) or exit code 2 (block) + stderr findings |

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

### 3.2 Tier1Label (enum)

Tier 1 direct identifier categories detected by redact-core. These values do not overlap with SpanLabel.

| Value | Description |
|-------|-------------|
| `NAME` | Person names |
| `EMAIL` | Email addresses |
| `PHONE` | Phone numbers |
| `ADDRESS` | Physical or mailing addresses |
| `GOV-ID` | Government-issued identifiers (SSN, passport, driver's license) |
| `FINANCIAL` | Financial account numbers (credit card, bank account) |
| `DOB` | Dates of birth |
| `BIOMETRIC` | Biometric identifiers |

### 3.3 Tier1Finding

A single Tier 1 detection result with span-level detail.

| Field | Type | Description |
|-------|------|-------------|
| `label` | Tier1Label | The category of direct identifier detected |
| `text` | string | The matched text |
| `start` | integer | Character offset of the match start |
| `end` | integer | Character offset of the match end (exclusive) |

### 3.4 RiskLevel (enum)

| Value | Description |
|-------|-------------|
| `LOW` | No meaningful re-identification risk |
| `MEDIUM` | Some quasi-identifiers present; limited re-identification risk in isolation |
| `HIGH` | Quasi-identifier combination sufficient to re-identify or locate an individual |

### 3.5 DetectionResult

The output of a single detection invocation. The model performs multilabel classification — it reports which quasi-identifier categories are present, not where they appear in the text.

| Field | Type | Description |
|-------|------|-------------|
| `labels` | set\<SpanLabel\> | Quasi-identifier categories present in the text (empty set if none) |
| `risk` | RiskLevel | Overall re-identification risk assessment |
| `rationale` | string | Brief explanation of quasi-identifier combinations (empty string if risk is LOW) |

### 3.6 Example

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

### 3.7 EvaluationReport

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
| Microsoft Phi-3 Mini (3.8B) | Training Pipeline, Detection Interface | Base model for QLoRA fine-tuning and inference |
| Faker library | Data Generation Pipeline | Synthetic data for template-based generation |
| Frontier LLM API (Claude / GPT-4) | Data Generation Pipeline | LLM-augmented example generation and auto-labeling |
| Label Studio | Human Review Workflow | Annotation interface for spot-checking |
| redact-core | PII Scanner Binary | Tier 1 direct identifier detection (pattern-based, Rust crate) |
| llama.cpp | PII Scanner Binary | GGUF model inference for Tier 2 quasi-identifier detection |

## 5. Key Constraints

- **Offline inference.** The Detection Interface and Claude Code Hooks make zero network calls after model loading. All inference is local.
- **Consumer hardware.** The merged model must run on CPU without GPU. Training requires a single GPU with 16GB VRAM.
- **No real PII.** No pipeline ingests, stores, or processes real personal data at any stage.
- **1,024 token limit.** Input text exceeding the model's context window (1,024 tokens) is truncated. The system does not implement sliding windows.

# QLoRA Fine-Tuning Methodology

Technical reference for the contextual PII tagger's data generation pipeline, training procedure, and inference strategy. Intended for a data scientist audience.

---

## 1. Data Generation Pipeline

The dataset is fully synthetic — no real personal information is used. Generation runs in five stages, producing a single combined pool that is stratified-shuffled before splitting.

### 1.1 Pipeline Overview

```
Stage 1: Template generation (Faker-filled YAML patterns)
Stage 2: LLM-augmented generation (Claude CLI, structured JSON)
    ↓
Combined pool → Stratified shuffle → 80/10/10 split
    ↓
Stage 3: Auto-labeling validation (second LLM pass)
Stage 4: Hard negative injection (per-split)
    ↓
Output: train.jsonl, validation.jsonl, test.jsonl
```

### 1.2 Default Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_count` | 50,000 | Target dataset size |
| `template_fraction` | 0.5 | Proportion from templates vs. LLM |
| `hard_negative_ratio` | 0.10 | Hard negatives as fraction of each split |
| `seed` | None | Random seed for reproducibility |
| `model` | sonnet | Claude model for LLM stages |

With defaults: 22,500 template examples + 22,500 LLM examples = 45,000 non-hard-negative examples, plus ~5,556 hard negatives injected post-split, yielding ~50,556 total.

### 1.3 Stage 1: Template-Based Generation

Templates are YAML files (one per domain) containing parameterized sentence patterns. Slot placeholders like `{NAME}`, `{HOSPITAL}`, `{CITY}` are filled by Faker with a deterministic seed. Each template carries pre-annotated labels and risk levels.

Round-robin domain selection and cyclic pattern selection guarantee balanced coverage. There are 40+ slot types covering names, locations, medical terms, workplaces, device identifiers, and temporal patterns.

**Strengths:** Deterministic, fast, predictable label distribution.
**Limitations:** Fixed patterns can produce formulaic text; relies on LLM stage for diversity.

### 1.4 Stage 2: LLM-Augmented Generation

Calls the Claude CLI to generate batches of structured JSON examples. Each call requests multiple examples (batch sizes are model-aware — see §1.7) and targets a single domain.

The prompt specifies:
- The full SpanLabel taxonomy (8 categories)
- Output schema (text, labels, risk, rationale, domain)
- Risk-level invariants
- Requirements for diversity and naturalism

Generation distributes across four domains in round-robin, with a 1.3× over-request multiplier to absorb parse failures. Failed batches are retried up to 5 consecutive rounds.

### 1.5 Stage 3: Auto-Labeling Validation

A second LLM pass reviews all generated examples in batches. For each example the validator may:
- Correct misassigned labels
- Adjust the risk level
- Generate or fix the rationale
- Flag the example as invalid (removed from dataset)

This two-pass approach catches systematic errors from Stage 2 — particularly label omissions and risk/rationale inconsistencies.

### 1.6 Stage 4: Hard Negative Injection

Hard negatives are texts that superficially resemble PII but are not: historical references, fictional characters, public figures in public contexts, generic statements, and hypothetical scenarios.

They are generated in a single LLM pass (batched), then distributed proportionally across splits so each split contains exactly `ratio / (1 - ratio) × existing_count` hard negatives. All hard negatives carry empty labels, LOW risk, and the `hard-negative` source tag.

### 1.7 Model-Aware Batch Limits

LLM calls are capped based on the model's max output token budget (with 20% headroom):

| Model | Output Tokens | Structured (§2) | Validation (§3) | Simple strings (§4) |
|-------|--------------|-----------------|-----------------|---------------------|
| Haiku (8,192) | 6,500 usable | ~86/call | ~81/call | ~216/call |
| Sonnet (16,384) | 13,000 usable | ~173/call | ~162/call | ~433/call |
| Opus (32,768) | 26,000 usable | ~346/call | ~325/call | ~866/call |

Token estimates: ~75 tokens per structured JSON example, ~80 per validation example (includes input echo), ~30 per simple string.

---

## 2. Randomization Strategy

### 2.1 Stratified Shuffle

Split assignment uses **stratified shuffling** over `(domain, risk)` strata (4 domains × 3 risk levels = 12 strata). Within each stratum, examples are independently shuffled and assigned to train/validation/test at 80/10/10 proportions. This guarantees that each split receives a proportional share of every domain × risk combination, rather than relying on statistical convergence from a uniform shuffle.

After split assignment, examples within each split are shuffled again to remove residual stratum ordering.

**Rounding:** Per-stratum rounding may shift the global split counts by a few examples relative to the exact 80/10/10 target. At 50k examples across 12 strata, the maximum deviation is ±12 examples (~0.02%).

### 2.2 Alternative Strategies Considered

| Strategy | Mechanism | Trade-off |
|----------|-----------|-----------|
| **Fisher-Yates (uniform)** | Shuffle full list, slice at 80%/90% boundaries | Simple; balanced in expectation but not guaranteed per stratum |
| **Stratified (chosen)** | Shuffle and split within each `(domain, risk)` stratum | Guarantees proportional representation; minor rounding artifact |
| **Reservoir sampling** | Stream-process examples into fixed-size reservoir | Useful when data doesn't fit in memory; unnecessary at 50k (~15 MB) |
| **Hash-based deterministic** | `hash(text + seed) % 10` assigns split | Re-runnable without full dataset in memory; no shuffle step; poor stratification |

### 2.3 Memory Considerations

Each example is ~300 bytes in memory (200B text + 100B metadata). At 50,000 examples the full dataset is ~15 MB — trivially fits in RAM. Reservoir sampling or streaming approaches are unnecessary below several million examples.

---

## 3. Taxonomy

Eight quasi-identifier categories (Tier 2 PII) that carry re-identification risk in combination:

| SpanLabel | Description |
|-----------|-------------|
| `LOCATION` | Geographic references (city, neighborhood, address) |
| `WORKPLACE` | Work context (company, department, job title) |
| `ROUTINE` | Recurring patterns (commute, schedule, appointments) |
| `MEDICAL-CONTEXT` | Health context (hospital, specialty, condition) |
| `DEMOGRAPHIC` | Personal attributes (age, ethnicity, religion) |
| `DEVICE-ID` | Device identifiers (model, serial number) |
| `CREDENTIAL` | Authentication hints (username, password hint) |
| `QUASI-ID` | Explicit quasi-identifier combinations |

Three ordinal risk levels:

| RiskLevel | Meaning | Rationale |
|-----------|---------|-----------|
| `LOW` | No PII or single low-risk quasi-identifier | Not provided (empty) |
| `MEDIUM` | Moderate re-identification risk | Required when 2+ labels present |
| `HIGH` | High re-identification risk from multiple categories | Required when 2+ labels present |

**Invariants enforced at every stage:**
1. Empty labels → risk must be LOW
2. Risk LOW → rationale must be empty
3. Risk MEDIUM/HIGH with 2+ labels → rationale must be non-empty

---

## 4. Training Configuration

### 4.1 Base Model

**Phi-3 Mini 4K Instruct** (3.8B parameters, MIT license). Instruction-tuned with a 4,096-token context window. Selected for its strong instruction-following at a parameter count that fits consumer hardware under QLoRA.

### 4.2 Quantization

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `load_in_4bit` | true | 4-bit weight quantization via bitsandbytes |
| `bnb_4bit_compute_dtype` | bfloat16 | Numeric stability during forward pass |
| `bnb_4bit_quant_type` | nf4 | Normal Float 4 — information-theoretically optimal for normally distributed weights |

Reduces base model memory from ~8 GB (fp16) to ~3 GB. **Note:** bitsandbytes requires CUDA. On Apple Silicon (MPS), the model loads in bf16 without quantization (~8 GB unified memory).

### 4.3 LoRA Adapter

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank (`r`) | 16 | Sufficient capacity for a multilabel classification task |
| Alpha (`α`) | 32 | 2× rank — standard scaling factor from LoRA literature |
| Dropout | 0.05 | Light regularization to prevent overfitting on synthetic data |
| Target modules | `q_proj`, `v_proj`, `k_proj`, `o_proj` | All self-attention projection matrices |
| Trainable parameters | ~8M (0.2% of base) | Highly parameter-efficient |
| Task type | CAUSAL_LM | Autoregressive language modeling objective |

The effective update to each attention weight is `ΔW = (α/r) × BA`, where `B ∈ ℝ^{d×r}` and `A ∈ ℝ^{r×d}`. With α/r = 2, the adapter contribution is scaled 2× relative to the low-rank product.

### 4.4 Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 3 | Sufficient for behavioral adaptation; low overfitting risk with 40k+ examples |
| Per-device batch size | 4 | Fits in 16 GB VRAM with QLoRA + bf16 |
| Gradient accumulation steps | 4 | Effective batch size = 16 |
| Learning rate | 2 × 10⁻⁴ | Standard for QLoRA; higher than full fine-tuning due to fewer trainable parameters |
| LR scheduler | Cosine | Smooth decay to zero; superior to linear for generalization |
| Warmup ratio | 0.05 | 5% of total steps (~375 of 7,500) to stabilize early gradients |
| Max sequence length | 1,024 tokens | Covers all prompt+completion pairs; oversize examples are skipped |
| Precision | bf16 | Mixed precision for throughput and stability |
| Logging steps | 10 | Frequent enough to monitor training loss trajectory |
| Save strategy | Per epoch | 3 checkpoints; final epoch used for adapter merge |
| Reporting | None | No external experiment tracker |

### 4.5 Effective Training Statistics

With 40,000 train examples (post-validation, pre-hard-negative), effective batch size 16, and 3 epochs:

| Metric | Value |
|--------|-------|
| Steps per epoch | ceil(40,000 / 16) = 2,500 |
| Total optimization steps | 7,500 |
| Warmup steps | 375 |
| Learning rate at step 375 | 2 × 10⁻⁴ (peak) |
| Learning rate at step 7,500 | ~0 (cosine minimum) |

---

## 5. Training Data Formatting

Each example is formatted as a Phi-3 chat prompt-completion pair:

```
<|user|>
Classify which quasi-identifier PII categories are present in the
following text. Return the list of category labels from the taxonomy,
an overall risk score (LOW/MEDIUM/HIGH), and a brief rationale.

Text: {example.text}
<|end|>
<|assistant|>
{"labels":["LOCATION","ROUTINE"],"risk":"MEDIUM","rationale":"..."}
<|end|>
```

- Completion JSON uses compact separators (no whitespace) for token efficiency
- Token budget is 1,024; examples exceeding this are **skipped**, not truncated
- Training set is shuffled after formatting via Fisher-Yates before being passed to SFTTrainer

---

## 6. Inference

### 6.1 Model Loading

The trained LoRA adapter is merged into the base model via `merge_and_unload()`, producing a standalone model that loads without the `peft` library. Alternatively, the adapter can be loaded on top of the base model at inference time.

### 6.2 Prompt Assembly

The inference prompt template is identical to the training template. User text is truncated to fit within a 1,024-token budget after accounting for the template overhead (~80–90 tokens), leaving ~930 tokens for input text.

### 6.3 Decoding

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Strategy | Greedy (`do_sample=False`) | Deterministic output for classification |
| Max new tokens | 256 | Upper bound on JSON completion length |
| Temperature | 1.0 | Overridden by greedy decoding |

The completion is parsed as JSON and validated against the DetectionResult invariants. Output is a `DetectionResult` containing the detected labels, risk level, and rationale.

---

## 7. Adapter Merging

Post-training, the LoRA adapter weights are merged into the base model:

1. Load the unquantized base model (Phi-3 Mini)
2. Load the LoRA adapter via `PeftModel.from_pretrained`
3. Call `merge_and_unload()` — absorbs `ΔW` into the base weights
4. Save the merged model and tokenizer

The merged model produces identical outputs to base+adapter but requires no adapter-aware loading path at inference time.

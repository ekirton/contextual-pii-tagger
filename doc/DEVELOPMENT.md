# Development Guide

Instructions for generating training data, running human review, fine-tuning the model, and evaluating results.

## Prerequisites

**Install uv** (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Python environment:**

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

**Ollama** is required for LLM-augmented dataset generation stages. Install it and pull the default model:

```bash
# Install Ollama: https://ollama.com/download
ollama pull qwen2.5:3b
```

**Hardware:**
- Dataset generation: Apple Silicon Mac with 16GB+ RAM (local inference via Ollama). 32GB recommended for 14B models.
- Training: GPU with 16GB+ VRAM (QLoRA reduces base model VRAM from ~8GB to ~3GB)
- Evaluation: CPU or GPU

## 1. Generate the Training Dataset

The pipeline produces 12,500 synthetic examples in an 80:10:10 train/validation/test split across five stages.

### Quick start (template-only, no LLM)

```bash
./scripts/generate-training-data.sh \
  --count 1000 \
  --seed 42 \
  --templates-only \
  --output ./data/
```

### Full pipeline (with LLM augmentation)

```bash
./scripts/generate-training-data.sh \
  --count 12500 \
  --seed 42 \
  --model qwen2.5:3b \
  --template-fraction 0.5 \
  --output ./data/
```

| Flag | Description | Default |
|------|-------------|---------|
| `-c, --count` | Total examples to generate | 1000 |
| `-s, --seed` | Random seed for reproducibility | none |
| `-o, --output` | Output directory | `data/` |
| `-m, --model` | Ollama model tag for LLM stages | `qwen2.5:3b` |
| `-t, --template-fraction` | Fraction from templates vs. LLM | 0.5 |
| `--templates-only` | Skip LLM stages | off |

### Pipeline stages

1. **Template generation** — Faker library fills parameterized templates across four domains (medical, scheduling, workplace, personal)
2. **LLM augmentation** — Local LLM (via Ollama) generates diverse, naturalistic examples targeting specific label distributions
3. **Auto-labeling** — Second LLM pass validates and corrects category labels, risk scores, and rationales on LLM-generated examples (template examples are passed through unchanged)
4. **Hard negative injection** — Adds 10% non-PII examples per split (historical references, public figures, generic statements)
5. **Human review** — 1% random sample selected for manual spot-checking (see next section)

### Output

Three JSONL files in the output directory:

```
data/train.jsonl       # ~10,000 examples
data/validation.jsonl  #  ~1,250 examples
data/test.jsonl        #  ~1,250 examples
```

Each line is a JSON Example record:

```json
{
  "id": "train-00001",
  "text": "I saw my pulmonologist at Johnson LLC Hospital last Sunday",
  "labels": ["MEDICAL-CONTEXT", "ROUTINE", "WORKPLACE"],
  "risk": "MEDIUM",
  "rationale": "Medical specialty and specific hospital narrow identification.",
  "is_hard_negative": false,
  "split": "train",
  "domain": "medical",
  "source": "template"
}
```

## 2. Human Review (Spot-Check)

Select a 1% sample for manual review:

```python
from contextual_pii_tagger.data.human_review import select_review_sample
from contextual_pii_tagger.data.dataset_io import read_dataset

dataset = read_dataset("./data/")
sample = select_review_sample(dataset, ratio=0.01, seed=42)
# ~200 examples across all splits
```

Export the sample to Label Studio or another annotation tool, review labels and risk scores, then apply corrections:

```python
from contextual_pii_tagger.data.human_review import Correction, apply_corrections
from contextual_pii_tagger.data.dataset_io import read_dataset, write_dataset

corrections = [
    Correction(id="train-00042", labels=frozenset(["LOCATION", "ROUTINE"])),
    Correction(id="test-00317", risk="HIGH"),
]

dataset = read_dataset("./data/")
corrected = apply_corrections(dataset, corrections)
write_dataset(corrected, "./data/")
```

Reviewers check:
- Are the category labels correct and complete?
- Does the risk score match the combination of labels?
- Are hard negatives truly non-sensitive?

## 3. Fine-Tune the Model

### Training configuration

The default config is at `src/contextual_pii_tagger/train/config.yaml`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | `microsoft/Phi-3-mini-4k-instruct` | 3.8B params |
| Quantization | 4-bit NF4 | Reduces VRAM to ~3GB |
| LoRA rank | 16 | ~8M trainable params |
| LoRA targets | q_proj, v_proj, k_proj, o_proj | Attention layers |
| Epochs | 3 | |
| Batch size | 4 (x4 gradient accumulation = 16 effective) | |
| Learning rate | 2e-4 (cosine schedule, 5% warmup) | |
| Max sequence length | 1,024 tokens | |

### Run training

```python
from contextual_pii_tagger.train.train import load_config, train
from contextual_pii_tagger.train.data_utils import prepare_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
dataset = prepare_dataset("./data/train.jsonl", tokenizer)

config = load_config("./src/contextual_pii_tagger/train/config.yaml")
train(config, dataset)
```

Output: LoRA adapter weights at `./output/contextual-pii-tagger/`.

### Merge the adapter

Create a standalone model that doesn't require `peft` at inference time:

```python
from contextual_pii_tagger.train.merge import merge_adapter

merge_adapter(
    base_model_path="microsoft/Phi-3-mini-4k-instruct",
    adapter_path="./output/contextual-pii-tagger",
    output_path="./output/merged-model",
)
```

## 4. Evaluate

### Run evaluation

```python
from contextual_pii_tagger.detector import PIIDetector
from contextual_pii_tagger.eval.evaluate import evaluate
from contextual_pii_tagger.data.dataset_io import read_dataset

detector = PIIDetector.from_pretrained("./output/merged-model")
all_data = read_dataset("./data/")
test_data = [ex for ex in all_data if ex.split == "test"]

report = evaluate(detector, test_data, model_name="fine-tuned")
```

### Compare against XGBoost baseline

```python
from contextual_pii_tagger.eval.baseline import train_baseline
from contextual_pii_tagger.eval.evaluate import evaluate, compare_models

train_data = [ex for ex in all_data if ex.split == "train"]

baseline = train_baseline(train_data)
baseline_report = evaluate(baseline, test_data, model_name="XGBoost")

comparison = compare_models(report, baseline_report)
```

### Success metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Multilabel F1 | >= 0.80 | Macro-average F1 across all 8 SpanLabel categories |
| Risk accuracy | >= 0.85 | Fraction of correct LOW/MEDIUM/HIGH predictions |
| False negative rate | <= 0.08 | Fraction of PII texts classified as clean |
| QUASI-ID F1 | >= 0.70 | F1 on the combination-detection label (hardest category) |
| Hard negative precision | >= 0.92 | Fraction of non-PII texts correctly passed |

## 5. Run Tests

```bash
uv run pytest test/ -x -q
```

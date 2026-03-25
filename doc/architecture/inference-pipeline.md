# Inference Pipeline

**Features:** F-01 (Quasi-Identifier Detection), F-04 (Detection Interface), F-11 (Scanner Binary)
**Requirements:** R-DET-01, R-DET-02, R-DET-03, R-DET-04, R-API-01, R-BIN-01

---

## 1. Pipeline Overview

The inference pipeline transforms raw text into a DetectionResult. It runs entirely offline after model loading.

```
                    ┌─────────────────────────────────────────────┐
                    │           Detection Interface               │
                    │                                             │
 Input text ──────> │  ┌──────────┐  ┌──────────┐  ┌──────────┐ │ ──────> DetectionResult
                    │  │  Prompt   │─>│  Model   │─>│  Output  │ │
                    │  │ Assembly  │  │ Inference│  │  Parser  │ │
                    │  └──────────┘  └──────────┘  └──────────┘ │
                    └─────────────────────────────────────────────┘
```

## 2. Components

### 2.1 Prompt Assembly

**Input:** Raw text string
**Output:** Tokenized prompt ready for model inference

Wraps the input text in the same instruction template used during training:

```
<|user|>
Classify which quasi-identifier PII categories are present in the
following text. Return the list of category labels from the taxonomy,
an overall risk score (LOW/MEDIUM/HIGH), and a brief rationale.

Text: {input_text}
<|end|>
<|assistant|>
```

The assembled prompt is tokenized using the base model's tokenizer. If the tokenized prompt exceeds 1,024 tokens, the input text is truncated to fit within the limit (prompt template tokens are reserved first).

### 2.2 Model Inference

**Input:** Tokenized prompt
**Output:** Raw model completion (token sequence)

The model generates a completion containing the JSON-formatted DetectionResult (category labels, risk level, rationale). Generation uses greedy decoding (temperature 0) for deterministic output.

The model is loaded once at initialization. Two loading modes are supported:
- **Merged weights:** Single model file, no `peft` dependency. Preferred for deployment.
- **Base + adapter:** Base model with LoRA adapter applied at load time. Used during development.

### 2.3 Output Parser

**Input:** Raw model completion string
**Output:** DetectionResult

Parses the model's JSON output into a structured DetectionResult. The parser handles:

- **Valid JSON:** Extracts labels set, RiskLevel, and rationale directly.
- **Malformed JSON:** Attempts repair (unclosed brackets, trailing commas). If repair fails, returns a DetectionResult with an empty label set, risk `LOW`, and a rationale indicating a parse failure.
- **Invalid SpanLabel:** Any label not in the SpanLabel enum is dropped with a warning.
- **Deduplication:** Duplicate labels in the model output are collapsed to a set.

## 3. Detection Interface (Python API)

The `PIIDetector` class is the primary entry point for application code.

### Initialization

```
PIIDetector.from_pretrained(model_path: string) -> PIIDetector
```

Loads the model from a local directory or HuggingFace model ID. After initialization, no network calls are made.

### Detection

```
PIIDetector.detect(text: string) -> DetectionResult
```

Runs the full inference pipeline (prompt assembly → model inference → output parsing) and returns a DetectionResult.

## 4. Performance Considerations

- **Model loading:** Loading the merged model takes several seconds on CPU. This is a one-time cost at startup.
- **Inference latency:** Single-example inference on CPU is expected to take 1-5 seconds depending on input length and hardware. This is acceptable for interactive use but not for high-throughput batch processing without optimization.
- **Memory:** The 4-bit quantized model requires approximately 3GB of RAM. The merged (non-quantized) model requires approximately 8GB.
- **Batch processing:** The Detection Interface processes one text at a time. Batch processing requires calling `detect()` in a loop. Parallelization is left to the caller.

## 5. Rust Implementation

The Tier 2 inference pipeline has a parallel Rust implementation within the PII Scanner Binary (see [rust-scanner.md](rust-scanner.md)). The Rust version uses llama.cpp with GGUF model format instead of PyTorch, but implements identical prompt assembly, greedy decoding, and output parsing logic. The Python pipeline remains the reference implementation for training and evaluation workflows.

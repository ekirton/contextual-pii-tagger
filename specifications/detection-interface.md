# Detection Interface Specification

**Architecture:** [inference-pipeline.md](../doc/architecture/inference-pipeline.md), Sections 2-3
**Features:** F-01, F-04

---

## 1. PIIDetector

The primary entry point for application code. Wraps model loading, inference, and output parsing.

### 1.1 from_pretrained

```
PIIDetector.from_pretrained(model_path: string) -> PIIDetector
```

**REQUIRES:**
- `model_path` is a non-empty string pointing to either:
  - A local directory containing merged model weights, or
  - A HuggingFace model ID (e.g., `"username/contextual-pii-tagger"`).
- The model at `model_path` is compatible with the Phi-3 Mini architecture.

**ENSURES:**
- Returns a PIIDetector instance with the model loaded and ready for inference.
- The tokenizer is loaded from the same `model_path`.
- No network calls are made after this method returns (if `model_path` is a local directory, no network calls are made at all; if it is a HuggingFace ID, the download occurs during this call only).

**RAISES:**
- `FileNotFoundError` if `model_path` is a local path that does not exist.
- `ValueError` if the model at `model_path` is not a compatible architecture.

### 1.2 detect

```
PIIDetector.detect(text: string) -> DetectionResult
```

**REQUIRES:**
- The PIIDetector has been initialized via `from_pretrained`.
- `text` is a non-empty string.

**ENSURES:**
- Returns a valid DetectionResult (satisfying all DetectionResult invariants from entities.md Section 3).
- No network calls are made.
- The method is stateless: calling `detect(text)` twice with the same input produces the same output.

**RAISES:**
- `ValueError` if `text` is empty.

**EDGE CASES:**
- If `text` tokenizes to more than 1,024 tokens (including the prompt template), the input is truncated. The DetectionResult reflects only the truncated portion. No error is raised.
- If the model produces malformed output, the Output Parser handles recovery (see output-parser.md). The caller always receives a valid DetectionResult.

---

## 2. Prompt Assembly

Internal component. Not part of the public API.

### 2.1 assemble_prompt

```
assemble_prompt(text: string, tokenizer: Tokenizer) -> TokenSequence
```

**REQUIRES:**
- `text` is a non-empty string.
- `tokenizer` is a loaded Phi-3 tokenizer.

**ENSURES:**
- Returns a token sequence formatted as:
  ```
  <|user|>
  Classify which quasi-identifier PII categories are present in the
  following text. Return the list of category labels from the taxonomy,
  an overall risk score (LOW/MEDIUM/HIGH), and a brief rationale.

  Text: {text}
  <|end|>
  <|assistant|>
  ```
- The total token count does not exceed 1,024.
- If `text` would cause the token count to exceed 1,024, `text` is truncated at a token boundary. The template tokens are reserved first; the remaining budget is allocated to `text`.

**MAINTAINS:**
- The prompt template is identical to the template used during training (training-pipeline.md Section 3).

---

## 3. Model Inference

Internal component. Not part of the public API.

### 3.1 generate

```
generate(prompt_tokens: TokenSequence, model: Model) -> string
```

**REQUIRES:**
- `prompt_tokens` is a valid token sequence from `assemble_prompt`.
- `model` is a loaded Phi-3 model (merged or base+adapter).

**ENSURES:**
- Returns the model's completion as a decoded string.
- Generation uses greedy decoding (temperature = 0).
- Output is deterministic: the same `prompt_tokens` always produces the same result.
- No network calls are made.

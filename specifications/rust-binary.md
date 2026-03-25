# Rust PII Scanner Binary Specification

**Architecture:** [rust-scanner.md](../doc/architecture/rust-scanner.md)
**Features:** F-10, F-11

---

## 1. CLI Entry Point

```
pii-scanner --tier1 --hook <hook_type>
pii-scanner --tier2 --hook <hook_type>
```

**REQUIRES:**
- Exactly one of `--tier1` or `--tier2` is specified.
- `--hook` value is one of: `user_prompt`, `pre_tool_use`, `post_tool_use`.
- `stdin` contains a JSON payload from Claude Code.
- For `--tier2`: a valid GGUF model is available at `PII_MODEL_PATH` or `~/.cache/contextual-pii-tagger/model.gguf`.

**ENSURES:**
- Exactly one of three exit codes is returned (see Section 4).
- On exit code 2, stderr contains JSON (format depends on tier — see Sections 2 and 3).
- On exit code 0, stderr is empty.
- On exit code 1, stderr contains a human-readable error message (not JSON).
- No output is written to stdout.

---

## 2. Tier 1 Scanning

```
scan_tier1(text: string) -> list<Tier1Finding>
```

Uses the `redact-core` crate to detect direct identifiers in the input text.

**REQUIRES:**
- `text` is a non-empty string.

**ENSURES:**
- Returns a list of `Tier1Finding` records (may be empty).
- Each finding has a valid `Tier1Label`, non-empty `text`, and valid character offsets.
- Findings are ordered by `start` offset (ascending).
- No network calls are made during detection.

### 2.1 Tier 1 Stderr Format (Exit 2)

```json
{"findings": [{"label": "EMAIL", "text": "john@acme.com", "start": 42, "end": 55}]}
```

**ENSURES:**
- The JSON object has a single key `findings` containing an array of Tier1Finding objects.
- Output is compact (no whitespace), single line.

---

## 3. Tier 2 Scanning

```
scan_tier2(text: string, model_path: string) -> DetectionResult
```

Runs the three-stage inference pipeline: prompt assembly, model inference, output parsing.

**REQUIRES:**
- `text` is a non-empty string.
- `model_path` points to a valid GGUF model file.

**ENSURES:**
- Returns a valid DetectionResult satisfying all invariants from entities.md Section 3.
- No network calls are made during detection.

### 3.1 Prompt Assembly

```
assemble_prompt(text: string, tokenizer: Tokenizer) -> Vec<token_id>
```

**REQUIRES:**
- `text` is a non-empty string.
- `tokenizer` is loaded from the GGUF model or a compatible tokenizer file.

**ENSURES:**
- The assembled prompt uses the identical template as the Python implementation (`src/contextual_pii_tagger/prompt.py`):
  ```
  <|user|>
  Classify which quasi-identifier PII categories are present in the
  following text. Return the list of category labels from the taxonomy,
  an overall risk score (LOW/MEDIUM/HIGH), and a brief rationale.

  Text: {text}
  <|end|>
  <|assistant|>
  ```
- The token sequence length does not exceed 1,024 tokens.
- If the full prompt exceeds 1,024 tokens, the input text is truncated at a token boundary.
- Template tokens are reserved first; remaining budget is allocated to the input text.

### 3.2 Model Inference

```
generate(prompt_tokens: Vec<token_id>, model: Model) -> string
```

**REQUIRES:**
- `prompt_tokens` is a non-empty token sequence of at most 1,024 tokens.
- `model` is a loaded llama.cpp model.

**ENSURES:**
- Output is generated with greedy decoding (temperature 0, no sampling).
- Maximum 256 new tokens are generated.
- The prompt tokens are stripped from the output; only the completion is returned.
- No network calls are made.

### 3.3 Output Parsing

```
parse_output(raw_output: string) -> DetectionResult
```

This is a total function — it never fails.

**REQUIRES:**
- `raw_output` is a string (possibly empty or malformed).

**ENSURES:**
- Returns a valid DetectionResult satisfying all invariants from entities.md Section 3.
- Implements the same parsing logic as `src/contextual_pii_tagger/output_parser.py`:
  1. JSON extraction: direct parse → markdown code fences → embedded JSON → repair → fallback.
  2. JSON repair: unclosed brackets/braces, trailing commas, single quotes → double quotes, unquoted keys.
  3. Field extraction with defaults: labels (list, default []), risk (string, default "LOW"), rationale (string, default "").
  4. Label validation: invalid SpanLabel values are dropped; duplicates collapsed.
  5. Consistency enforcement: empty labels → LOW risk, empty rationale; LOW risk → empty rationale; MEDIUM/HIGH with 2+ labels and empty rationale → "Multiple quasi-identifiers detected."
- If all parsing and repair attempts fail, returns: `DetectionResult(labels={}, risk=LOW, rationale="")`.

### 3.4 Tier 2 Stderr Format (Exit 2)

Identical to the Python hook script — a JSON-serialized DetectionResult:

```json
{"labels": ["WORKPLACE", "ROUTINE"], "risk": "MEDIUM", "rationale": "Workplace and routine..."}
```

**ENSURES:**
- The JSON output satisfies all DetectionResult invariants from entities.md Section 3.
- `labels` is a sorted array of strings.
- Output is compact (no whitespace), single line.

---

## 4. Exit Code Contract

| Exit Code | Condition | stderr Content |
|-----------|-----------|----------------|
| 0 | No PII detected, or no text to scan | Empty |
| 2 | PII detected | JSON (format per tier — Section 2.1 or 3.4) |
| 1 | Any error during processing | Human-readable error message |

### 4.1 Exit 0 — Pass

**REQUIRES:**
- `extract_text` returned an empty string, OR
- Tier 1: `scan_tier1` returned an empty list, OR
- Tier 2: `scan_tier2` returned a DetectionResult with empty labels.

**ENSURES:**
- Exit code is 0.
- stderr is empty.

### 4.2 Exit 2 — Block

**REQUIRES:**
- Tier 1: `scan_tier1` returned a non-empty list, OR
- Tier 2: `scan_tier2` returned a DetectionResult with non-empty labels.

**ENSURES:**
- Exit code is 2.
- stderr contains the appropriate JSON format (compact, single line).

### 4.3 Exit 1 — Error (Fail-Open)

**REQUIRES:**
- An error occurred during payload parsing, model loading, or inference.

**ENSURES:**
- Exit code is 1.
- stderr contains a human-readable error message.

---

## 5. Payload Extraction

```
extract_text(hook_type: string, payload: dict) -> string
```

Identical contract to hook-script.md Section 2.1.

**ENSURES:**
- For `user_prompt`: returns `payload["query"]`.
- For `pre_tool_use`: JSON-serializes `payload["tool_input"]`.
- For `post_tool_use`: returns `payload["tool_output"]`.
- Returns an empty string if the expected field is missing or empty.

---

## 6. Model Loading

**REQUIRES:**
- For Tier 2: `PII_MODEL_PATH` is set to a valid `.gguf` file path, OR `~/.cache/contextual-pii-tagger/model.gguf` exists.

**ENSURES:**
- The GGUF model is loaded via memory-mapped I/O (mmap).
- If the model cannot be loaded, exit code 1 is returned.

**MAINTAINS:**
- The model is loaded fresh on each invocation (hooks are stateless).
- Tier 1 requires no model loading.

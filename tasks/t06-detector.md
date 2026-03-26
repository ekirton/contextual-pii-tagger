# T-06: PIIDetector

**File:** `src/contextual_pii_tagger/detector.py`
**Spec:** [entities.md](../specifications/entities.md) §3
**Depends on:** T-01 (entities.py), T-03 (output_parser.py), T-05 (prompt.py)

## Scope

Implement the `PIIDetector` class — the primary public API.

## Deliverables

1. **PIIDetector.from_pretrained(model_path)** — Load model and tokenizer. Support local path and HuggingFace ID. Two modes: merged weights (no peft) and base+adapter.

2. **PIIDetector.detect(text)** — Full inference pipeline:
   - Validate text non-empty (raise ValueError)
   - Call `assemble_prompt(text, tokenizer)` for tokenization + truncation
   - Run model inference with greedy decoding (temperature=0)
   - Call `parse_output(raw_output, text)` to get DetectionResult
   - Return DetectionResult

3. **Internal generate function** — Greedy decoding, deterministic output.

## Acceptance Criteria

- `from_pretrained` raises FileNotFoundError / ValueError per spec.
- `detect` raises ValueError on empty text.
- `detect` is deterministic (same input → same output).
- No network calls after `from_pretrained` returns.
- All returned DetectionResults satisfy entity invariants.

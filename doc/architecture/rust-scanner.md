# Rust PII Scanner Binary

**Features:** F-10 (Tier 1 Detection), F-11 (Unified Scanner Binary)
**Requirements:** R-T1-01, R-BIN-01, R-BIN-02

---

## 1. Overview

The PII scanner binary (`pii-scanner`) is a compiled Rust executable that replaces the Python inference and hook layer for runtime use. It combines two detection tiers in a single binary, invoked with separate flags:

- `--tier1` — direct identifier detection via redact-core (sub-millisecond, pattern + NER)
- `--tier2` — quasi-identifier detection via llama.cpp (GGUF model, CPU inference)

```
                        ┌──────────────────────────────────────┐
  stdin (JSON) ────────>│           pii-scanner                │
                        │                                      │
                        │  ┌──────────────┐                    │
                        │  │   Payload    │                    │
                        │  │  Extraction  │                    │
                        │  └──────┬───────┘                    │
                        │         │                            │
                        │    ┌────┴────┐                       │
                        │    │ --tier? │                       │
                        │    └────┬────┘                       │
                        │   ┌────┘└────┐                       │
                        │   v          v                       │
                        │ ┌──────┐  ┌──────────────────────┐  │
                        │ │Tier 1│  │       Tier 2         │  │
                        │ │ NER  │  │ Prompt → LLM → Parse │  │
                        │ └──┬───┘  └──────────┬───────────┘  │
                        │    v                 v               │
                        │  Tier1Findings    DetectionResult    │
                        │    │                 │               │
                        │    v                 v               │
                        │  exit code + stderr JSON             │
                        └──────────────────────────────────────┘
```

## 2. Tier 1 Component

Uses the `redact-core` Rust crate for direct identifier detection.

**Entity mapping:**

| redact-core Entity | Tier1Label |
|-------------------|------------|
| Person name | `NAME` |
| Email address | `EMAIL` |
| Phone number | `PHONE` |
| Physical/mailing address | `ADDRESS` |
| SSN, passport, driver's license | `GOV-ID` |
| Credit card, bank account | `FINANCIAL` |
| Date of birth | `DOB` |
| Biometric identifier | `BIOMETRIC` |

**Output:** A list of `Tier1Finding` records, each containing the label, matched text, and character offsets (start, end) into the original input.

**Performance:** Sub-millisecond. Pattern-based detection with optional transformer NER. No model loading required for pattern-only mode.

## 3. Tier 2 Component

Uses `llama-cpp-rs` (Rust bindings for llama.cpp) to run the fine-tuned Phi-3 Mini model in GGUF format.

**Three-stage pipeline (mirrors Python inference pipeline):**

1. **Prompt Assembly.** Uses the identical template as `src/contextual_pii_tagger/prompt.py`. Tokenizes via the HuggingFace `tokenizers` crate. Truncates input text at token boundary if the assembled prompt exceeds 1,024 tokens.

2. **Model Inference.** Loads the GGUF model from `PII_MODEL_PATH` (or default `~/.cache/contextual-pii-tagger/model.gguf`). Greedy decoding, `max_new_tokens=256`. Model is memory-mapped for fast loading (~200ms vs. multi-second PyTorch load).

3. **Output Parser.** Identical logic to `src/contextual_pii_tagger/output_parser.py`: JSON extraction with repair (unclosed brackets, trailing commas, single quotes, unquoted keys), field extraction with defaults, label validation, and consistency enforcement. Total function — never fails, always returns a valid DetectionResult.

**Output:** A `DetectionResult` (labels, risk, rationale) — the same JSON format as the Python implementation.

## 4. Exit Code Contract

Identical to the Python hook script (see hook-integration.md Section 4):

| Exit Code | Tier 1 | Tier 2 |
|-----------|--------|--------|
| 0 | No direct identifiers found | No quasi-identifiers found |
| 2 | Direct identifiers found; stderr = Tier1Findings JSON | Quasi-identifiers found; stderr = DetectionResult JSON |
| 1 | Error during detection (fail-open) | Error during model load or inference (fail-open) |

Each tier writes its own JSON format to stderr. The formats are distinct and do not need to be merged — Claude Code runs them as separate hooks.

## 5. Hook Execution Order

Claude Code runs hooks sequentially within each event. The recommended configuration places Tier 1 before Tier 2:

1. **Tier 1 runs first** (sub-millisecond). If direct identifiers are found → exit 2, action blocked. Tier 2 does not run.
2. **Tier 2 runs second** (2-8 seconds). Only reached if Tier 1 passed. Scans for quasi-identifier combinations.

This ordering minimizes latency: obvious PII is caught instantly, and expensive model inference is skipped when unnecessary.

## 6. Model Conversion

The Tier 2 model must be converted from safetensors to GGUF format after training and merging:

```bash
# Convert merged safetensors model to GGUF
python llama.cpp/convert_hf_to_gguf.py \
    ~/.cache/contextual-pii-tagger/merged/ \
    --outfile ~/.cache/contextual-pii-tagger/model.gguf \
    --outtype q4_k_m
```

**Quantization:** Q4_K_M provides a good balance of model size (~2 GB) and accuracy for the 3.8B parameter Phi-3 Mini. Q5_K_M is an alternative if accuracy is prioritized over size.

## 7. Performance Budget

| Phase | Tier 1 | Tier 2 |
|-------|--------|--------|
| Binary startup | < 5ms | < 5ms |
| Stdin parsing | < 1ms | < 1ms |
| Model load | N/A | ~200ms (mmap) |
| Inference | < 1ms | 2-6s (CPU, Q4_K_M) |
| Output formatting | < 1ms | < 1ms |
| **Total** | **< 10ms** | **2-7s** |

Both tiers complete well within the 10-second Claude Code hook timeout.

## 8. Project Structure

```
rust/
├── Cargo.toml
├── src/
│   ├── main.rs           # CLI entry point (clap), stdin, exit codes
│   ├── payload.rs         # extract_text per hook type
│   ├── tier1.rs           # redact-core integration, Tier1Label, Tier1Finding
│   ├── tier2/
│   │   ├── mod.rs         # scan_tier2 orchestration
│   │   ├── prompt.rs      # prompt template, tokenization, truncation
│   │   ├── inference.rs   # llama.cpp model load + generate
│   │   └── parser.rs      # JSON extraction, repair, validation
│   ├── entities.rs        # SpanLabel, RiskLevel, DetectionResult
│   └── scan.rs            # top-level scan dispatch (tier1 vs tier2)
└── tests/
    ├── tier1_integration.rs
    ├── tier2_integration.rs
    └── exit_code_contract.rs
```

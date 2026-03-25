# F-11: Unified PII Scanner Binary

**Priority:** P1
**Requirements:** R-BIN-01, R-BIN-02

## What This Feature Does

Packages the Tier 1 direct identifier detector and the Tier 2 quasi-identifier detector into a single compiled Rust binary (`pii-scanner`). The binary replaces the Python inference and hook layer for runtime use, eliminating the need for Python, pip, PyTorch, or HuggingFace Transformers on the end user's machine.

The binary operates in two modes:

- `pii-scanner --tier1 --hook <event>` — runs Tier 1 detection (redact-core)
- `pii-scanner --tier2 --hook <event>` — runs Tier 2 detection (fine-tuned Phi-3 via llama.cpp)

Both modes read a JSON payload from stdin, scan the extracted text, and communicate results via exit codes and stderr — the same contract as the existing Python hook script.

## Why It Exists

The Python inference stack requires installing PyTorch (~2 GB), HuggingFace Transformers, and optionally PEFT — a heavy dependency chain for what should be a lightweight privacy gate. Cold-start latency (Python interpreter + torch import + model load) approaches the 10-second Claude Code hook timeout, leaving little room for actual inference.

A compiled binary with mmap-based model loading starts in milliseconds and runs inference without framework overhead. Distribution is a single file download rather than a pip install with platform-specific dependencies.

## Design Tradeoffs

- **GGUF model format.** The Tier 2 model must be converted from safetensors to GGUF format for llama.cpp. This adds a one-time conversion step after training but enables mmap loading and built-in quantization.
- **Two modes, not two binaries.** Tier 1 and Tier 2 share a single binary to simplify distribution. The `--tier1` / `--tier2` flag selects the detection mode. They run as separate Claude Code hooks so each can block independently.
- **Python package remains.** The Rust binary replaces only the inference and hook layer. Training, evaluation, and data generation continue to use the Python package. The Python `PIIDetector` class remains available for development and scripting workflows.

## What This Feature Does Not Provide

- A Python API replacement (the Python PIIDetector class remains for training and evaluation).
- GPU-accelerated inference (CPU only via llama.cpp).
- Streaming or batch-mode scanning.
- Cross-compilation for all platforms (initially targets the developer's build platform).

## Acceptance Criteria

### AC-01: Single binary, no Python runtime
**GIVEN** a machine with the `pii-scanner` binary and a GGUF model file
**WHEN** the binary is invoked
**THEN** it runs without Python, pip, torch, or transformers installed
*(Traces to R-BIN-01)*

### AC-02: Tier 1 mode
**GIVEN** the binary is invoked with `--tier1 --hook <event>`
**WHEN** a JSON payload is provided on stdin
**THEN** the binary scans for direct identifiers and returns exit 0 (clean) or exit 2 (PII found, findings on stderr)
*(Traces to R-BIN-01, R-T1-01)*

### AC-03: Tier 2 mode
**GIVEN** the binary is invoked with `--tier2 --hook <event>` and PII_MODEL_PATH points to a valid GGUF model
**WHEN** a JSON payload is provided on stdin
**THEN** the binary scans for quasi-identifier PII and returns exit 0 (clean) or exit 2 (PII found, DetectionResult JSON on stderr)
*(Traces to R-BIN-01)*

### AC-04: Exit code contract preserved
**GIVEN** any invocation of the binary
**WHEN** the scan completes or encounters an error
**THEN** the exit code follows the hook contract: 0 (pass), 2 (block), 1 (error/fail-open)
*(Traces to R-BIN-02)*

### AC-05: Environment variable interface preserved
**GIVEN** the binary is invoked in Tier 2 mode
**WHEN** PII_MODEL_PATH is set
**THEN** the model is loaded from that path; if unset, the default path is used
*(Traces to R-BIN-02)*

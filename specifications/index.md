# Specifications

Detailed specifications for each component, using Design by Contract (REQUIRES/ENSURES/MAINTAINS). Each spec traces to architecture documents in [doc/architecture/](../doc/architecture/index.md).

## Entity Specifications

| Document | Description |
|----------|-------------|
| [entities.md](entities.md) | SpanLabel, RiskLevel, DetectionResult, Example, EvaluationReport |

## Runtime Component Specifications

| Document | Description |
|----------|-------------|
| [detection-interface.md](detection-interface.md) | PIIDetector class: loading and detection |
| [output-parser.md](output-parser.md) | Model output parsing, JSON repair, label validation |
| [hook-script.md](hook-script.md) | Claude Code hook: payload extraction, exit codes (Python) |
| [rust-binary.md](rust-binary.md) | Rust PII scanner binary: Tier 1 + Tier 2 detection, CLI contract |

## Pipeline Specifications

| Document | Description |
|----------|-------------|
| [data-generation.md](data-generation.md) | Five-stage synthetic dataset generation |
| [training.md](training.md) | Example formatting and QLoRA fine-tuning |
| [evaluation.md](evaluation.md) | Metric computation, multilabel comparison, baseline comparison |

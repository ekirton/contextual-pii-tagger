# Specifications

Detailed specifications for each component, using Design by Contract (REQUIRES/ENSURES/MAINTAINS). Each spec traces to architecture documents in [doc/architecture/](../doc/architecture/index.md).

## Entity Specifications

| Document | Description |
|----------|-------------|
| [entities.md](entities.md) | SpanLabel, RiskLevel, DetectionResult, Example, EvaluationReport |

## Pipeline Specifications

| Document | Description |
|----------|-------------|
| [data-generation.md](data-generation.md) | Five-stage synthetic dataset generation |
| [training.md](training.md) | Example formatting and QLoRA fine-tuning |
| [evaluation.md](evaluation.md) | Metric computation, multilabel comparison, baseline comparison |

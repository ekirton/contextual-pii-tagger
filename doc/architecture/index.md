# Architecture

Documents in this directory describe how the system is built — pipelines, data flows, component responsibilities, and boundary contracts. Each document traces to features in [doc/features/](../features/index.md).

| Document | Description |
|----------|-------------|
| [system-overview.md](system-overview.md) | High-level system architecture, component boundaries, and entity definitions |
| [data-generation-pipeline.md](data-generation-pipeline.md) | Five-stage synthetic dataset generation pipeline |
| [training-pipeline.md](training-pipeline.md) | QLoRA fine-tuning pipeline and configuration |
| [evaluation-pipeline.md](evaluation-pipeline.md) | Evaluation against metrics and XGBoost baseline |

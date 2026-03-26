# contextual-pii-tagger

A research project investigating whether a small, locally-run model can detect **contextual and quasi-identifier PII** in free text — the kind of personally identifiable information that regex tools and NER systems miss entirely.

## The Problem

In 2000, Latanya Sweeney showed that **87% of the U.S. population** can be uniquely identified from just three pieces of information: zip code, date of birth, and gender. No name, no email, no SSN — just ordinary demographic details that, in combination, single out one person. Her study linked a public voter roll to "anonymized" hospital records and re-identified the governor of Massachusetts, demonstrating that quasi-identifiers are a real and practical privacy threat.

Existing PII tools (Presidio, spaCy, cloud APIs) catch direct identifiers like names, emails, and SSNs. They are blind to this kind of contextual PII — combinations of seemingly harmless details that together re-identify a person:

> *"The only female cardiologist at St. Mary's in Tucson who does pediatric cases on Thursdays."*

No name appears, yet this uniquely identifies someone. Every PII detector on the market would pass this text as clean. This project investigates whether a small model can detect that kind of risk.

## Method

- **Approach:** QLoRA fine-tuned Phi-3 Mini (3.8B) compared against an XGBoost classifier over spaCy embeddings
- **Training data:** 12,500 synthetically generated examples with human spot-checking
- **Scope:** Tier 2 quasi-identifiers only (see [PII Tier Classification](doc/pii-tiers.md))

## Evaluation Results

A QLoRA fine-tuned Phi-3 Mini (3.8B) was compared against an XGBoost baseline on 1,243 held-out test examples. The two models performed within noise of each other:

| Metric | LoRA | XGBoost | Delta |
|--------|------|---------|-------|
| Multilabel F1 | 0.8333 | 0.8347 | −0.0014 |
| Risk accuracy | 0.9316 | 0.9292 | +0.0024 |
| False negative rate | 0.0000 | 0.0206 | −0.0206 |
| QUASI-ID F1 | 0.5798 | 0.5809 | −0.0011 |
| Hard negative precision | 1.0000 | 0.9677 | +0.0323 |

The LoRA model showed a small advantage on false negatives and hard negative precision, but did not meaningfully outperform the baseline on the primary F1 metric. Given the comparable accuracy, XGBoost is the simpler and more practical approach — faster to train, no GPU requirement.

Full per-label results are in [`data/comparison-report.txt`](data/comparison-report.txt). The LoRA fine-tuning methodology is documented in [lora-fine-tuning.md](doc/lora-fine-tuning.md).

## Project Status

Research complete. This project ended at the evaluation comparing the QLoRA fine-tuned model against the XGBoost baseline. There is no installable deliverable.

The original motivation was to build a privacy gate for Claude Code sessions, but the Claude desktop app does not support hooks the way the CLI does. This project demonstrates that a fast, local model can flag quasi-identifier PII, but shipping it as a product would require either a hooks-capable chat interface or native platform support.

## Documentation

| Directory | Contents |
|-----------|----------|
| [features/](doc/features/) | Feature descriptions and acceptance criteria |
| [architecture/](doc/architecture/) | System design and pipeline documentation |
| [background/](doc/background/) | Background research and literature survey |

Specifications live in [`/specifications`](specifications/) and task breakdowns in [`/tasks`](tasks/).

## Development

See [DEVELOPMENT.md](doc/DEVELOPMENT.md) for instructions on generating the training dataset, running human review, fine-tuning the model, and evaluating results.

## References

- Sweeney, L. (2000). *Simple Demographics Often Identify People Uniquely.* Data Privacy Working Paper No. 3, Carnegie Mellon University. https://doi.org/10.1184/R1/6625769

## License

[MIT](LICENSE)

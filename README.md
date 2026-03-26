# contextual-pii-tagger

Detects **contextual and quasi-identifier PII** in free text — the kind of personally identifiable information that regex tools and NER systems miss entirely.

## The Problem

In 2000, Latanya Sweeney showed that **87% of the U.S. population** can be uniquely identified from just three pieces of information: zip code, date of birth, and gender. No name, no email, no SSN — just ordinary demographic details that, in combination, single out one person. Her study linked a public voter roll to "anonymized" hospital records and re-identified the governor of Massachusetts, demonstrating that quasi-identifiers are a real and practical privacy threat.

Existing PII tools (Presidio, spaCy, cloud APIs) catch direct identifiers like names, emails, and SSNs. They are blind to this kind of contextual PII — combinations of seemingly harmless details that together re-identify a person:

> *"The only female cardiologist at St. Mary's in Tucson who does pediatric cases on Thursdays."*

No name appears, yet this uniquely identifies someone. Every PII detector on the market would pass this text as clean. This project detects that kind of risk.

## Method

- **Model:** XGBoost classifier over spaCy embeddings — chosen after evaluation showed it matches a QLoRA fine-tuned LM at a fraction of the cost and complexity (see [Evaluation Results](#evaluation-results))
- **Training data:** 50,000 synthetically generated examples with human spot-checking
- **Scope:** Tier 2 quasi-identifiers (custom model); Tier 1 direct identifiers (via redact-core); sensitive context (Tier 3) is out of scope (see [PII Tier Classification](pii-tiers.md))

## Quick Example

```
$ echo '{"query":"I am the only female cardiologist at St Marys in Tucson"}' \
    | pii-scanner --tier2 --hook user_prompt
```

If quasi-identifier PII is detected, the binary exits with code 2 and writes the findings to stderr:

```json
{"labels":["DEMOGRAPHIC","WORKPLACE"],"risk":"HIGH","rationale":"Gender, specialty, and institution uniquely identify an individual."}
```

Clean text exits with code 0 and empty stderr.

## Evaluation Results

A QLoRA fine-tuned Phi-3 Mini (3.8B) was compared against an XGBoost baseline on 1,243 held-out test examples. The two models performed within noise of each other:

| Metric | LoRA | XGBoost | Delta |
|--------|------|---------|-------|
| Multilabel F1 | 0.8333 | 0.8347 | −0.0014 |
| Risk accuracy | 0.9316 | 0.9292 | +0.0024 |
| False negative rate | 0.0000 | 0.0206 | −0.0206 |
| QUASI-ID F1 | 0.5798 | 0.5809 | −0.0011 |
| Hard negative precision | 1.0000 | 0.9677 | +0.0323 |

The LoRA model showed a small advantage on false negatives and hard negative precision, but did not meaningfully outperform the baseline on the primary F1 metric. Given the comparable accuracy, XGBoost is the preferred production model — it is simpler to train, faster at inference, and requires no GPU.

Full per-label results are in [`data/comparison-report.txt`](../data/comparison-report.txt). The LoRA fine-tuning methodology is documented in [lora-fine-tuning.md](lora-fine-tuning.md).

## Deliverables

| Artifact | Description |
|----------|-------------|
| XGBoost model | Trained classifier for Tier 2 quasi-identifier detection |
| Benchmark dataset | 5,000-example human-reviewed test set |
| Rust scanner binary | `pii-scanner` — combined Tier 1 + Tier 2 detection, no Python required |
| Claude Code hooks | Privacy gate for prompt submission, tool use, and tool output |
| Interactive demo | HuggingFace Space |
| Training walkthrough | Colab notebook |

## Installation

Users need the `pii-scanner` binary and Claude Code hook configuration. No Python runtime or GPU is required.

### 1. Install system prerequisites

You need a [Rust toolchain](https://rustup.rs/) (1.75+).

### 2. Build the binary

```bash
cd rust
cargo build --release
```

The binary is at `rust/target/release/pii-scanner`. Copy it somewhere on your `PATH`.

### 3. Enable Claude Code hooks

Add the following to `.claude/settings.json` (or `.claude/settings.local.json`). Two hooks per event — Tier 1 runs first (fast, pattern-based), Tier 2 runs second (model inference):

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          { "type": "command", "command": "pii-scanner --tier1 --hook user_prompt" },
          { "type": "command", "command": "pii-scanner --tier2 --hook user_prompt", "timeout": 10000 }
        ]
      }
    ],
    "PreToolUse": [
      {
        "hooks": [
          { "type": "command", "command": "pii-scanner --tier1 --hook pre_tool_use" },
          { "type": "command", "command": "pii-scanner --tier2 --hook pre_tool_use", "timeout": 10000 }
        ]
      }
    ],
    "PostToolUse": [
      {
        "hooks": [
          { "type": "command", "command": "pii-scanner --tier1 --hook post_tool_use" },
          { "type": "command", "command": "pii-scanner --tier2 --hook post_tool_use", "timeout": 10000 }
        ]
      }
    ]
  }
}
```

### How hooks work

The PII scanner intercepts three points in your Claude Code session:

- **UserPromptSubmit** — scans your prompt before it is sent
- **PreToolUse** — scans tool arguments before a tool runs
- **PostToolUse** — scans tool output before it is returned to Claude

When PII is detected, the hook blocks the action and Claude presents the findings conversationally, offering options to redact, revise, or continue.

## Project Status

Proof of concept — pre-release. See the [backlog](requirements/backlog.md) for current priorities.

## Documentation

| Directory | Contents |
|-----------|----------|
| [requirements/](requirements/) | Product requirements and backlog |
| [features/](features/) | Feature descriptions and acceptance criteria |
| [architecture/](architecture/) | System design and pipeline documentation |
| [background/](background/) | Background research and literature survey |

Specifications live in [`/specifications`](../specifications/) and task breakdowns in [`/tasks`](../tasks/).

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for instructions on generating the training dataset, running human review, fine-tuning the model, and evaluating results.

## References

- Sweeney, L. (2000). *Simple Demographics Often Identify People Uniquely.* Data Privacy Working Paper No. 3, Carnegie Mellon University. https://doi.org/10.1184/R1/6625769

## License

[MIT](../LICENSE)

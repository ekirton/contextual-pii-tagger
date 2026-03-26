# Implementation Tasks

Decomposition of specifications into implementable units. Tasks are ordered by dependency — earlier tasks have no unmet dependencies; later tasks build on earlier ones.

## Task Status Key

- `[ ]` — Not started
- `[~]` — In progress
- `[x]` — Complete

## Phase 1: Core Entities (no dependencies)

| Task | File | Spec | Status |
|------|------|------|--------|
| [T-01](t01-entities.md) | `src/contextual_pii_tagger/entities.py` | entities.md §1-3 | [x] |
| [T-02](t02-example.md) | `src/contextual_pii_tagger/example.py` | entities.md §4-5 | [x] |

## Phase 2: Inference Components (depends on Phase 1)

| Task | File | Spec | Status |
|------|------|------|--------|
| [T-03](t03-output-parser.md) | `src/contextual_pii_tagger/output_parser.py` | entities.md §3 | [x] |
| [T-05](t05-prompt-assembly.md) | `src/contextual_pii_tagger/prompt.py` | training.md §1 | [x] |

## Phase 3: Detection Interface (depends on Phase 2)

| Task | File | Spec | Status |
|------|------|------|--------|
| [T-06](t06-detector.md) | `src/contextual_pii_tagger/detector.py` | entities.md §3 | [x] |

## Phase 4: Data Generation Pipeline (depends on Phase 1)

| Task | File | Spec | Status |
|------|------|------|--------|
| [T-09](t09-templates.md) | `src/contextual_pii_tagger/data/templates.py` | data-generation.md §2 | [x] |
| [T-10](t10-llm-generation.md) | `src/contextual_pii_tagger/data/llm_generate.py` | data-generation.md §3 | [x] |
| [T-11](t11-auto-labeling.md) | `src/contextual_pii_tagger/data/validate_labels.py` | data-generation.md §4 | [x] |
| [T-12](t12-hard-negatives.md) | `src/contextual_pii_tagger/data/hard_negatives.py` | data-generation.md §5 | [x] |
| [T-13](t13-dataset-io.md) | `src/contextual_pii_tagger/data/dataset_io.py` | data-generation.md §7 | [x] |
| [T-14](t14-generate-pipeline.md) | `src/contextual_pii_tagger/data/generate.py` | data-generation.md §1 | [x] |

## Phase 5: Training Pipeline (depends on Phases 1, 4)

| Task | File | Spec | Status |
|------|------|------|--------|
| [T-15](t15-training-format.md) | `src/contextual_pii_tagger/train/data_utils.py` | training.md §1-2 | [x] |
| [T-16](t16-training-loop.md) | `src/contextual_pii_tagger/train/train.py` | training.md §3-4 | [x] |
| [T-17](t17-merge-adapter.md) | `src/contextual_pii_tagger/train/merge.py` | training.md §4 | [x] |

## Phase 6: Evaluation Pipeline (depends on Phases 1, 3)

| Task | File | Spec | Status |
|------|------|------|--------|
| [T-19](t19-metrics.md) | `src/contextual_pii_tagger/eval/metrics.py` | evaluation.md §2-6 | [x] |
| [T-20](t20-xgboost-baseline.md) | `src/contextual_pii_tagger/eval/baseline.py` | evaluation.md §8 | [x] |
| [T-21](t21-evaluate-pipeline.md) | `src/contextual_pii_tagger/eval/evaluate.py` | evaluation.md §1, §7 | [x] |

## Phase 7: Packaging & Config

| Task | File | Spec | Status |
|------|------|------|--------|
| [T-22](t22-package-init.md) | `src/contextual_pii_tagger/__init__.py` + `pyproject.toml` | — | [x] |
| [T-23](t23-template-files.md) | `src/contextual_pii_tagger/data/templates/*.yaml` | data-generation.md §2.2 | [x] |
| [T-24](t24-training-config.md) | `src/contextual_pii_tagger/train/config.yaml` | training.md §3 | [x] |

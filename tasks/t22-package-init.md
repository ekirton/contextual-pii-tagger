# T-22: Package Initialization

**Files:** `src/contextual_pii_tagger/__init__.py`, `pyproject.toml`
**Spec:** — (packaging, no spec)
**Depends on:** T-01 through T-08 (all runtime components)

## Scope

Set up the Python package structure and public API exports.

## Deliverables

1. **`__init__.py`** — Export public API:
   - `PIIDetector` (from detector.py)
   - `DetectionResult`, `SpanLabel`, `RiskLevel` (from entities.py)

2. **`pyproject.toml`** — Package metadata, dependencies, entry points:
   - Dependencies: transformers, peft, bitsandbytes, accelerate, trl, datasets, fastapi, uvicorn, faker, xgboost, scikit-learn, spacy
   - Entry points for CLI commands (train, merge, evaluate, generate, serve)

3. **Package `__init__.py` files** — For subpackages: `data/`, `train/`, `eval/`, `hooks/`.

## Acceptance Criteria

- `from contextual_pii_tagger import PIIDetector` works.
- `pip install -e .` installs the package.
- All subpackage imports resolve.

# T-10: LLM-Augmented Generation

**File:** `src/contextual_pii_tagger/data/llm_generate.py`
**Spec:** [data-generation.md](../specifications/data-generation.md) §3
**Depends on:** T-01 (entities.py), T-02 (example.py)

## Scope

Implement `generate_from_llm(taxonomy, target_distribution, count) -> list[Example]`.

## Deliverables

1. **Generation prompt** — Construct prompts providing the Tier 2 taxonomy, target domain/label, and instructions for structured output (text + category labels JSON).

2. **Batch generation** — Generate in batches targeting specific domain/SpanLabel combinations for coverage.

3. **Response parsing** — Parse LLM responses into Example records. Handle malformed responses (retry or skip).

4. **Diversity controls** — Distribute batches across domains and SpanLabels per target_distribution.

## Acceptance Criteria

- All returned Examples have `source: llm-augmented`.
- Domain/SpanLabel distribution approximately matches target.
- All Example invariants hold.
- Generation prompt explicitly requests synthetic content only.

# T-05: Prompt Assembly

**File:** `src/contextual_pii_tagger/prompt.py`
**Spec:** [training.md](../specifications/training.md) §1
**Depends on:** T-01 (entities.py)

## Scope

Implement `assemble_prompt(text, tokenizer) -> TokenSequence` and the shared prompt template constant.

## Deliverables

1. **PROMPT_TEMPLATE constant** — The instruction template used by both training and inference. Single source of truth.

2. **assemble_prompt function** — Wrap input text in the template, tokenize, and truncate if needed:
   - Compute template token budget (tokenize template without text)
   - Remaining budget = 1024 - template tokens
   - If text tokens exceed budget, truncate at token boundary
   - Return complete token sequence

3. **get_template_text function** — Returns the filled template as a string (for training formatting).

## Acceptance Criteria

- Template matches the exact format from training-pipeline.md §3.
- Token count never exceeds 1,024.
- Truncation happens at token boundaries (no partial tokens).
- Template tokens are reserved first; text gets remaining budget.

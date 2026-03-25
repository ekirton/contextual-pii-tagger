"""Prompt assembly: shared template and tokenization for training and inference.

Spec: specifications/detection-interface.md §2
"""

from __future__ import annotations

PROMPT_TEMPLATE = (
    "<|user|>\n"
    "Classify which quasi-identifier PII categories are present in the\n"
    "following text. Return the list of category labels from the taxonomy,\n"
    "an overall risk score (LOW/MEDIUM/HIGH), and a brief rationale.\n"
    "\n"
    "Text: {text}\n"
    "<|end|>\n"
    "<|assistant|>\n"
)

MAX_SEQUENCE_LENGTH = 1024


def get_template_text(text: str) -> str:
    """Return the prompt template with *text* filled in."""
    return PROMPT_TEMPLATE.format(text=text)


def assemble_prompt(text: str, tokenizer: object) -> list[int]:
    """Tokenize *text* inside the prompt template, enforcing the 1,024 token limit.

    Template tokens are reserved first; the remaining budget is allocated
    to *text*.  If *text* would exceed the budget it is truncated at a
    token boundary.
    """
    # Compute template overhead (tokenize with empty text)
    template_without_text = get_template_text("")
    template_tokens: list[int] = tokenizer.encode(
        template_without_text, add_special_tokens=False
    )
    template_token_count = len(template_tokens)

    text_budget = MAX_SEQUENCE_LENGTH - template_token_count
    if text_budget <= 0:
        return template_tokens[:MAX_SEQUENCE_LENGTH]

    # Tokenize the raw text
    text_tokens: list[int] = tokenizer.encode(text, add_special_tokens=False)

    # Truncate text tokens to budget
    if len(text_tokens) > text_budget:
        text_tokens = text_tokens[:text_budget]

    # Reconstruct truncated text and re-tokenize the full prompt
    truncated_text = tokenizer.decode(text_tokens, skip_special_tokens=True)
    full_prompt = get_template_text(truncated_text)
    final_tokens: list[int] = tokenizer.encode(
        full_prompt, add_special_tokens=False
    )

    # Safety clamp (re-tokenization might shift counts slightly)
    if len(final_tokens) > MAX_SEQUENCE_LENGTH:
        final_tokens = final_tokens[:MAX_SEQUENCE_LENGTH]

    return final_tokens

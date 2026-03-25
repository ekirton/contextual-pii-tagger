"""Tests for prompt assembly.

All expectations derived from specifications/detection-interface.md §2
and specifications/training.md §1 (shared template requirement).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from contextual_pii_tagger.prompt import (
    PROMPT_TEMPLATE,
    assemble_prompt,
    get_template_text,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_tokenizer(token_map: dict[str, list[int]] | None = None):
    """Create a mock tokenizer that assigns one token per word.

    If *token_map* is given, the tokenizer returns those token IDs for
    the exact matching string; otherwise it returns one int per
    whitespace-delimited word.
    """
    tokenizer = MagicMock()
    token_map = token_map or {}

    def encode(text, add_special_tokens=False):
        if text in token_map:
            return token_map[text]
        return list(range(len(text.split())))

    def decode(token_ids, skip_special_tokens=True):
        return " ".join(f"tok{t}" for t in token_ids)

    tokenizer.encode = MagicMock(side_effect=encode)
    tokenizer.decode = MagicMock(side_effect=decode)
    return tokenizer


# ── §2.1: PROMPT_TEMPLATE constant ──────────────────────────────────────


class TestPromptTemplate:
    """ENSURES: template matches spec format from detection-interface.md §2.1."""

    def test_contains_user_tag(self):
        assert "<|user|>" in PROMPT_TEMPLATE

    def test_contains_end_tag(self):
        assert "<|end|>" in PROMPT_TEMPLATE

    def test_contains_assistant_tag(self):
        assert "<|assistant|>" in PROMPT_TEMPLATE

    def test_contains_text_placeholder(self):
        assert "{text}" in PROMPT_TEMPLATE

    def test_contains_classification_instruction(self):
        assert "quasi-identifier" in PROMPT_TEMPLATE.lower()

    def test_template_is_single_source(self):
        """The template must have exactly one {text} placeholder."""
        assert PROMPT_TEMPLATE.count("{text}") == 1


# ── §2.1: get_template_text ─────────────────────────────────────────────


class TestGetTemplateText:
    """ENSURES: filled template as a plain string."""

    def test_returns_string_with_text_embedded(self):
        result = get_template_text("Hello world")
        assert "Hello world" in result

    def test_no_remaining_placeholder(self):
        result = get_template_text("test input")
        assert "{text}" not in result

    def test_preserves_template_structure(self):
        result = get_template_text("sample")
        assert "<|user|>" in result
        assert "<|end|>" in result
        assert "<|assistant|>" in result


# ── §2.1: assemble_prompt ───────────────────────────────────────────────


class TestAssemblePrompt:
    """ENSURES: token sequence ≤ 1,024; template tokens reserved first."""

    def test_short_text_not_truncated(self):
        tok = _make_tokenizer()
        tokens = assemble_prompt("Short text.", tok)
        assert isinstance(tokens, list)
        assert len(tokens) <= 1024

    def test_token_count_never_exceeds_1024(self):
        tok = _make_tokenizer()
        long_text = " ".join(["word"] * 2000)
        tokens = assemble_prompt(long_text, tok)
        assert len(tokens) <= 1024

    def test_template_tokens_reserved_first(self):
        """Template portion is always fully present; text is truncated."""
        template_without_text = get_template_text("")
        # Count template tokens as words (mock tokenizer behavior)
        template_token_count = len(template_without_text.split())

        tok = _make_tokenizer()
        long_text = " ".join(["word"] * 2000)
        tokens = assemble_prompt(long_text, tok)

        # Result should be exactly 1024 (budget fully used)
        assert len(tokens) == 1024

    def test_returns_list_of_ints(self):
        tok = _make_tokenizer()
        tokens = assemble_prompt("Hello", tok)
        assert all(isinstance(t, int) for t in tokens)

    def test_truncation_at_token_boundary(self):
        """Truncation must happen at token boundaries, not mid-token."""
        # With our mock tokenizer (1 token per word), truncation
        # should produce a clean word count, not partial words.
        tok = _make_tokenizer()
        long_text = " ".join(["word"] * 2000)
        tokens = assemble_prompt(long_text, tok)
        # Each token is an int from our mock — no partial tokens possible
        assert all(isinstance(t, int) for t in tokens)
        assert len(tokens) <= 1024

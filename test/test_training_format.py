"""Tests for training data formatting.

All expectations derived from specifications/training.md §1-2.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from contextual_pii_tagger.entities import RiskLevel, SpanLabel
from contextual_pii_tagger.example import Example
from contextual_pii_tagger.prompt import PROMPT_TEMPLATE
from contextual_pii_tagger.train.data_utils import format_example, prepare_dataset


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_tokenizer():
    """Mock tokenizer: one token per whitespace-delimited word."""
    tok = MagicMock()
    tok.encode = MagicMock(side_effect=lambda text, **kw: list(range(len(text.split()))))
    tok.decode = MagicMock(side_effect=lambda ids, **kw: " ".join(f"w{i}" for i in ids))
    tok.eos_token_id = 2
    tok.pad_token_id = 0
    return tok


def _make_example(
    text="I work at Acme Corp near downtown.",
    labels=None,
    risk=RiskLevel.MEDIUM,
    rationale="Workplace and location combined.",
    is_hard_negative=False,
    split="train",
    domain="workplace",
    example_id=None,
):
    if labels is None:
        labels = frozenset({SpanLabel.WORKPLACE, SpanLabel.LOCATION})
    if example_id is None:
        example_id = f"{split}-00001"
    return Example(
        id=example_id,
        text=text,
        labels=labels,
        risk=risk,
        rationale=rationale,
        is_hard_negative=is_hard_negative,
        split=split,
        domain=domain,
        source="hard-negative" if is_hard_negative else "template",
    )


# ── §1: format_example ──────────────────────────────────────────────────


class TestFormatExample:
    """ENSURES: prompt-completion pair from Example."""

    def test_returns_dict_with_text_key(self):
        tok = _make_tokenizer()
        result = format_example(_make_example(), tok)
        assert result is not None
        assert "text" in result

    def test_prompt_uses_shared_template(self):
        """MAINTAINS: prompt template identical to inference template."""
        tok = _make_tokenizer()
        result = format_example(_make_example(), tok)
        # The formatted text should contain the template's instruction
        assert "quasi-identifier" in result["text"].lower()
        assert "<|user|>" in result["text"]

    def test_completion_is_compact_json(self):
        tok = _make_tokenizer()
        result = format_example(_make_example(), tok)
        text = result["text"]
        # Extract the assistant's response
        assistant_marker = "<|assistant|>"
        idx = text.index(assistant_marker) + len(assistant_marker)
        completion = text[idx:].strip()
        # Remove trailing end token if present
        if "<|end|>" in completion:
            completion = completion[: completion.index("<|end|>")].strip()
        parsed = json.loads(completion)
        assert "labels" in parsed
        assert "risk" in parsed
        assert "rationale" in parsed

    def test_hard_negative_formatted_correctly(self):
        """ENSURES: hard negatives → empty labels, LOW risk, empty rationale."""
        tok = _make_tokenizer()
        hn = _make_example(
            text="The Battle of Gettysburg was important.",
            labels=frozenset(),
            risk=RiskLevel.LOW,
            rationale="",
            is_hard_negative=True,
        )
        result = format_example(hn, tok)
        text = result["text"]
        assistant_marker = "<|assistant|>"
        idx = text.index(assistant_marker) + len(assistant_marker)
        completion = text[idx:].strip()
        if "<|end|>" in completion:
            completion = completion[: completion.index("<|end|>")].strip()
        parsed = json.loads(completion)
        assert parsed["labels"] == []
        assert parsed["risk"] == "LOW"
        assert parsed["rationale"] == ""

    def test_skips_example_exceeding_1024_tokens(self):
        """ENSURES: examples > 1,024 tokens are skipped (returns None)."""
        tok = _make_tokenizer()
        long_text = " ".join(["word"] * 2000)
        ex = _make_example(text=long_text)
        result = format_example(ex, tok)
        assert result is None

    def test_skipped_example_logged(self, caplog):
        """ENSURES: skipped examples produce a warning."""
        tok = _make_tokenizer()
        long_text = " ".join(["word"] * 2000)
        ex = _make_example(text=long_text)
        with caplog.at_level(logging.WARNING):
            format_example(ex, tok)
        assert any("skip" in r.message.lower() for r in caplog.records)

    def test_labels_sorted_in_completion(self):
        """Labels in the completion JSON should be sorted for consistency."""
        tok = _make_tokenizer()
        ex = _make_example(
            labels=frozenset({SpanLabel.WORKPLACE, SpanLabel.LOCATION}),
        )
        result = format_example(ex, tok)
        text = result["text"]
        assistant_marker = "<|assistant|>"
        idx = text.index(assistant_marker) + len(assistant_marker)
        completion = text[idx:].strip()
        if "<|end|>" in completion:
            completion = completion[: completion.index("<|end|>")].strip()
        parsed = json.loads(completion)
        assert parsed["labels"] == sorted(parsed["labels"])


# ── §2: prepare_dataset ─────────────────────────────────────────────────


class TestPrepareDataset:
    """ENSURES: loads train.jsonl, formats, excludes oversize, shuffles."""

    def test_loads_and_formats(self, tmp_path):
        tok = _make_tokenizer()
        ex = _make_example()
        train_file = tmp_path / "train.jsonl"
        train_file.write_text(json.dumps(ex.to_dict()) + "\n")

        dataset = prepare_dataset(str(tmp_path), tok)
        assert len(dataset) == 1

    def test_excludes_oversize_examples(self, tmp_path):
        tok = _make_tokenizer()
        short = _make_example(text="Short text.", example_id="train-00001")
        long_text = " ".join(["word"] * 2000)
        long_ex = _make_example(text=long_text, example_id="train-00002")

        train_file = tmp_path / "train.jsonl"
        lines = [json.dumps(short.to_dict()), json.dumps(long_ex.to_dict())]
        train_file.write_text("\n".join(lines) + "\n")

        dataset = prepare_dataset(str(tmp_path), tok)
        assert len(dataset) == 1

    def test_returns_list_of_dicts(self, tmp_path):
        tok = _make_tokenizer()
        ex = _make_example()
        train_file = tmp_path / "train.jsonl"
        train_file.write_text(json.dumps(ex.to_dict()) + "\n")

        dataset = prepare_dataset(str(tmp_path), tok)
        assert all(isinstance(d, dict) for d in dataset)
        assert all("text" in d for d in dataset)

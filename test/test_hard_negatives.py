"""Tests for hard negative injection (Stage 4).

All expectations derived from specifications/data-generation.md §5.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from contextual_pii_tagger.entities import RiskLevel, SpanLabel
from contextual_pii_tagger.example import Example
from contextual_pii_tagger.data.hard_negatives import (
    inject_hard_negatives,
    build_hard_negative_prompt,
    parse_hard_negative_response,
    compute_hard_negative_counts,
)


def _make_example(split: str, num: int) -> Example:
    return Example(
        id=f"{split}-{num:05d}",
        text=f"Example text for {split} {num}",
        labels=frozenset({SpanLabel.LOCATION}),
        risk=RiskLevel.MEDIUM,
        rationale="",
        is_hard_negative=False,
        split=split,
        domain="medical",
        source="template",
    )


def _make_dataset(train: int = 80, val: int = 10, test: int = 10) -> list[Example]:
    examples = []
    for i in range(1, train + 1):
        examples.append(_make_example("train", i))
    for i in range(1, val + 1):
        examples.append(_make_example("validation", i))
    for i in range(1, test + 1):
        examples.append(_make_example("test", i))
    return examples


def _mock_hard_negative_texts(count: int) -> list[str]:
    return [f"George Washington crossed the Delaware in 1776 (example {i})." for i in range(count)]


def _mock_ollama_response(texts: list[str]) -> MagicMock:
    body = json.dumps({
        "message": {"role": "assistant", "content": json.dumps(texts)},
    }).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestComputeHardNegativeCounts:
    """Tests for computing per-split hard negative counts."""

    def test_basic_10_percent(self):
        counts = compute_hard_negative_counts(
            {"train": 90, "validation": 9, "test": 9}, ratio=0.10
        )
        # train: need hn where hn/(90+hn) = 0.10 → hn = 10
        assert counts["train"] == 10
        # validation: hn/(9+hn) = 0.10 → hn = 1
        assert counts["validation"] == 1
        assert counts["test"] == 1

    def test_zero_ratio(self):
        counts = compute_hard_negative_counts(
            {"train": 80, "validation": 10, "test": 10}, ratio=0.0
        )
        assert all(c == 0 for c in counts.values())


class TestBuildHardNegativePrompt:
    """Tests for prompt construction."""

    def test_specifies_count(self):
        prompt = build_hard_negative_prompt(count=10)
        assert "10" in prompt

    def test_requests_non_pii(self):
        prompt = build_hard_negative_prompt(count=5)
        assert "not" in prompt.lower() and ("pii" in prompt.lower() or "personally" in prompt.lower())

    def test_requests_json(self):
        prompt = build_hard_negative_prompt(count=5)
        assert "JSON" in prompt or "json" in prompt


class TestParseHardNegativeResponse:
    """Tests for parsing hard negative texts."""

    def test_parses_text_list(self):
        texts = _mock_hard_negative_texts(3)
        results = parse_hard_negative_response(texts)
        assert len(results) == 3
        assert all(isinstance(t, str) for t in results)

    def test_skips_empty_strings(self):
        texts = ["Valid text.", "", "Another valid text."]
        results = parse_hard_negative_response(texts)
        assert len(results) == 2


class TestInjectHardNegatives:
    """Integration tests with mocked Ollama."""

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_adds_correct_count(self, mock_urlopen):
        examples = _make_dataset(train=90, val=9, test=9)
        # Need 10 train + 1 val + 1 test = 12 hard negatives
        mock_urlopen.return_value = _mock_ollama_response(
            _mock_hard_negative_texts(15)
        )
        result = inject_hard_negatives(examples, ratio=0.10, model="qwen2.5:7b")
        total_hn = sum(1 for ex in result if ex.is_hard_negative)
        assert total_hn == 12

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_hard_negatives_have_correct_fields(self, mock_urlopen):
        examples = _make_dataset(train=9, val=0, test=0)
        mock_urlopen.return_value = _mock_ollama_response(_mock_hard_negative_texts(5))
        result = inject_hard_negatives(examples, ratio=0.10, model="qwen2.5:7b")
        for ex in result:
            if ex.is_hard_negative:
                assert ex.labels == frozenset()
                assert ex.risk == RiskLevel.LOW
                assert ex.rationale == ""
                assert ex.source == "hard-negative"

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_hard_negatives_per_split(self, mock_urlopen):
        examples = _make_dataset(train=90, val=9, test=9)
        mock_urlopen.return_value = _mock_ollama_response(_mock_hard_negative_texts(15))
        result = inject_hard_negatives(examples, ratio=0.10, model="qwen2.5:7b")

        by_split: dict[str, list] = {"train": [], "validation": [], "test": []}
        for ex in result:
            if ex.is_hard_negative:
                by_split[ex.split].append(ex)

        assert len(by_split["train"]) == 10
        assert len(by_split["validation"]) == 1
        assert len(by_split["test"]) == 1

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_ids_follow_pattern(self, mock_urlopen):
        examples = _make_dataset(train=9, val=0, test=0)
        mock_urlopen.return_value = _mock_ollama_response(_mock_hard_negative_texts(5))
        result = inject_hard_negatives(examples, ratio=0.10, model="qwen2.5:7b")
        for ex in result:
            if ex.is_hard_negative:
                assert ex.id.startswith("train-")
                # ID should be after existing examples
                num = int(ex.id.split("-")[1])
                assert num > 9

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_preserves_original_examples(self, mock_urlopen):
        examples = _make_dataset(train=9, val=0, test=0)
        mock_urlopen.return_value = _mock_ollama_response(_mock_hard_negative_texts(5))
        result = inject_hard_negatives(examples, ratio=0.10, model="qwen2.5:7b")
        original_ids = {ex.id for ex in examples}
        result_ids = {ex.id for ex in result if not ex.is_hard_negative}
        assert original_ids == result_ids

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_zero_ratio_adds_nothing(self, mock_urlopen):
        examples = _make_dataset(train=10, val=1, test=1)
        result = inject_hard_negatives(examples, ratio=0.0, model="qwen2.5:7b")
        assert len(result) == len(examples)
        assert not mock_urlopen.called

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_connection_failure_raises(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        with pytest.raises(RuntimeError, match="Ollama"):
            inject_hard_negatives(_make_dataset(train=9), ratio=0.10, model="qwen2.5:7b")

"""Tests for auto-labeling validation (Stage 3).

All expectations derived from specifications/data-generation.md §4.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from contextual_pii_tagger.entities import RiskLevel, SpanLabel
from contextual_pii_tagger.example import Example
from contextual_pii_tagger.data.validate_labels import (
    validate_labels,
    build_validation_prompt,
    parse_validation_response,
)


def _make_example(
    num: int = 1,
    labels: frozenset[SpanLabel] = frozenset({SpanLabel.LOCATION}),
    risk: RiskLevel = RiskLevel.MEDIUM,
    rationale: str = "",
    source: str = "llm-augmented",
) -> Example:
    return Example(
        id=f"train-{num:05d}",
        text=f"Example text number {num} with contextual details.",
        labels=labels,
        risk=risk,
        rationale=rationale,
        is_hard_negative=False,
        split="train",
        domain="medical",
        source=source,
    )


def _mock_validation_response(examples: list[Example], corrections: dict | None = None) -> list[dict]:
    """Build a mock validation response. corrections maps index to overrides."""
    corrections = corrections or {}
    result = []
    for i, ex in enumerate(examples):
        overrides = corrections.get(i, {})
        result.append({
            "id": ex.id,
            "labels": overrides.get("labels", sorted(l.value for l in ex.labels)),
            "risk": overrides.get("risk", ex.risk.value),
            "rationale": overrides.get("rationale", ex.rationale),
            "valid": overrides.get("valid", True),
        })
    return result


def _mock_ollama_response(payload: list[dict]) -> MagicMock:
    body = json.dumps({
        "message": {"role": "assistant", "content": json.dumps(payload)},
    }).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestBuildValidationPrompt:
    """Tests for prompt construction."""

    def test_contains_example_text(self):
        ex = _make_example()
        prompt = build_validation_prompt([ex])
        assert ex.text in prompt

    def test_contains_taxonomy(self):
        prompt = build_validation_prompt([_make_example()])
        for label in SpanLabel:
            assert label.value in prompt

    def test_requests_json(self):
        prompt = build_validation_prompt([_make_example()])
        assert "JSON" in prompt or "json" in prompt


class TestParseValidationResponse:
    """Tests for parsing validation results."""

    def test_keeps_valid_examples(self):
        examples = [_make_example(1), _make_example(2)]
        response = _mock_validation_response(examples)
        results = parse_validation_response(examples, response)
        assert len(results) == 2

    def test_removes_invalid_examples(self):
        examples = [_make_example(1), _make_example(2)]
        response = _mock_validation_response(examples, {1: {"valid": False}})
        results = parse_validation_response(examples, response)
        assert len(results) == 1
        assert results[0].id == "train-00001"

    def test_applies_label_correction(self):
        examples = [_make_example(1)]
        response = _mock_validation_response(
            examples,
            {0: {"labels": ["LOCATION", "ROUTINE"], "risk": "MEDIUM",
                 "rationale": "Location and routine narrow identity."}},
        )
        results = parse_validation_response(examples, response)
        assert SpanLabel.ROUTINE in results[0].labels
        assert SpanLabel.LOCATION in results[0].labels

    def test_applies_risk_correction(self):
        examples = [_make_example(
            1,
            labels=frozenset({SpanLabel.LOCATION, SpanLabel.ROUTINE}),
            risk=RiskLevel.MEDIUM,
            rationale="Some rationale.",
        )]
        response = _mock_validation_response(
            examples,
            {0: {"risk": "HIGH", "rationale": "Stronger risk assessment."}},
        )
        results = parse_validation_response(examples, response)
        assert results[0].risk == RiskLevel.HIGH

    def test_generates_rationale(self):
        """Validation can add a label and provide rationale."""
        examples = [_make_example(
            1,
            labels=frozenset({SpanLabel.LOCATION}),
            risk=RiskLevel.MEDIUM,
            rationale="",
        )]
        response = _mock_validation_response(
            examples,
            {0: {"labels": ["LOCATION", "ROUTINE"], "risk": "MEDIUM",
                 "rationale": "Generated rationale for medium risk."}},
        )
        results = parse_validation_response(examples, response)
        assert results[0].rationale == "Generated rationale for medium risk."
        assert SpanLabel.ROUTINE in results[0].labels

    def test_preserves_id_and_split(self):
        examples = [_make_example(1)]
        response = _mock_validation_response(examples)
        results = parse_validation_response(examples, response)
        assert results[0].id == examples[0].id
        assert results[0].split == examples[0].split

    def test_positional_fallback_when_all_ids_hallucinated(self):
        """When LLM returns wrong IDs for all entries, fall back to positional matching (§4.1)."""
        examples = [_make_example(42), _make_example(43)]
        # LLM hallucinates sequential IDs instead of actual IDs
        response = [
            {"id": "train-00001", "labels": ["LOCATION"], "risk": "MEDIUM",
             "rationale": "", "valid": True},
            {"id": "train-00002", "labels": ["LOCATION"], "risk": "MEDIUM",
             "rationale": "", "valid": True},
        ]
        results = parse_validation_response(examples, response)
        assert len(results) == 2
        assert results[0].id == "train-00042"
        assert results[1].id == "train-00043"

    def test_positional_fallback_respects_valid_flag(self):
        """Positional fallback still honors valid=false to remove examples."""
        examples = [_make_example(10), _make_example(11)]
        response = [
            {"id": "wrong-00001", "labels": ["LOCATION"], "risk": "MEDIUM",
             "rationale": "", "valid": True},
            {"id": "wrong-00002", "labels": ["LOCATION"], "risk": "MEDIUM",
             "rationale": "", "valid": False},
        ]
        results = parse_validation_response(examples, response)
        assert len(results) == 1
        assert results[0].id == "train-00010"

    def test_no_positional_fallback_when_some_ids_match(self):
        """Positional fallback only activates when ALL IDs fail to match."""
        examples = [_make_example(1), _make_example(2), _make_example(3)]
        response = [
            {"id": "train-00001", "labels": ["LOCATION"], "risk": "MEDIUM",
             "rationale": "", "valid": True},
            {"id": "wrong-id", "labels": ["LOCATION"], "risk": "MEDIUM",
             "rationale": "", "valid": True},
            {"id": "train-00003", "labels": ["LOCATION"], "risk": "MEDIUM",
             "rationale": "", "valid": True},
        ]
        results = parse_validation_response(examples, response)
        # Only the 2 with matching IDs should survive; no positional fallback
        assert len(results) == 2

    def test_no_positional_fallback_when_length_mismatch(self):
        """Positional fallback requires response length == input length."""
        examples = [_make_example(42), _make_example(43)]
        response = [
            {"id": "wrong-00001", "labels": ["LOCATION"], "risk": "MEDIUM",
             "rationale": "", "valid": True},
        ]
        results = parse_validation_response(examples, response)
        assert len(results) == 0

    def test_all_results_satisfy_invariants(self):
        examples = [
            _make_example(1, frozenset({SpanLabel.LOCATION}), RiskLevel.MEDIUM, ""),
            _make_example(2, frozenset({SpanLabel.WORKPLACE, SpanLabel.LOCATION}),
                          RiskLevel.MEDIUM, "Work and location."),
        ]
        response = _mock_validation_response(examples)
        results = parse_validation_response(examples, response)
        for r in results:
            assert isinstance(r, Example)


class TestValidateLabels:
    """Integration tests with mocked Ollama."""

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_returns_validated_examples(self, mock_urlopen):
        examples = [_make_example(1), _make_example(2)]
        mock_urlopen.return_value = _mock_ollama_response(
            _mock_validation_response(examples)
        )
        results = validate_labels(examples, model="qwen2.5:3b")
        assert len(results) == 2

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_retries_failed_batch(self, mock_urlopen):
        """A transient failure must be retried and succeed on next attempt (§3.2)."""
        import urllib.error
        examples = [_make_example(1), _make_example(2)]
        good_resp = _mock_ollama_response(_mock_validation_response(examples))
        mock_urlopen.side_effect = [
            urllib.error.URLError("Connection refused"),
            good_resp,
        ]
        results = validate_labels(examples, model="qwen2.5:3b")
        assert len(results) == 2

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_raises_after_max_consecutive_failures(self, mock_urlopen):
        """After consecutive failures with no progress, raise RuntimeError (§3.2)."""
        import urllib.error
        examples = [_make_example(1)]
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        with pytest.raises(RuntimeError, match="consecutive|failed"):
            validate_labels(examples, model="qwen2.5:3b")

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_empty_result_retries_batch(self, mock_urlopen):
        """An empty result field triggers a retry, not a silent skip (§3.2)."""
        examples = [_make_example(1)]
        empty_body = json.dumps({
            "message": {"role": "assistant", "content": ""},
        }).encode()
        empty_resp = MagicMock()
        empty_resp.read.return_value = empty_body
        empty_resp.__enter__ = lambda s: s
        empty_resp.__exit__ = MagicMock(return_value=False)
        good_resp = _mock_ollama_response(_mock_validation_response(examples))
        mock_urlopen.side_effect = [empty_resp, good_resp]
        results = validate_labels(examples, model="qwen2.5:3b")
        assert len(results) == 1

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_empty_input_returns_empty(self, mock_urlopen):
        results = validate_labels([], model="qwen2.5:3b")
        assert results == []
        assert not mock_urlopen.called

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_template_examples_skip_validation(self, mock_urlopen):
        """Template-sourced examples must pass through unchanged (§4.1)."""
        template_ex = _make_example(1, source="template")
        llm_ex = _make_example(2, source="llm-augmented")
        mock_urlopen.return_value = _mock_ollama_response(
            _mock_validation_response([llm_ex])
        )
        results = validate_labels([template_ex, llm_ex], model="qwen2.5:3b")
        # Both should be in results
        assert len(results) == 2
        # Template example unchanged
        template_result = next(r for r in results if r.source == "template")
        assert template_result.id == template_ex.id
        assert template_result.labels == template_ex.labels

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_only_template_examples_no_llm_call(self, mock_urlopen):
        """If all examples are template-sourced, no LLM call should be made."""
        examples = [_make_example(1, source="template"), _make_example(2, source="template")]
        results = validate_labels(examples, model="qwen2.5:3b")
        assert len(results) == 2
        assert not mock_urlopen.called

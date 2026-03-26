"""Tests for LLM-augmented generation (Stage 2).

All expectations derived from specifications/data-generation.md §3.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from contextual_pii_tagger.entities import RiskLevel, SpanLabel
from contextual_pii_tagger.data.raw_example import RawExample
from contextual_pii_tagger.data.llm_generate import (
    generate_from_llm,
    build_generation_prompt,
    parse_llm_response,
)


def _mock_llm_response(count: int = 5) -> list[dict]:
    """Build a realistic LLM response payload."""
    examples = []
    domains = ["medical", "scheduling", "workplace", "personal"]
    label_sets = [
        (["MEDICAL-CONTEXT", "WORKPLACE"], "MEDIUM", "Doctor and hospital narrow identity."),
        (["ROUTINE", "LOCATION"], "MEDIUM", "Route and time reveal commute."),
        (["WORKPLACE", "DEMOGRAPHIC"], "MEDIUM", "Role and team narrow identity."),
        (["LOCATION", "ROUTINE", "DEMOGRAPHIC"], "HIGH", "Religion and neighborhood combine."),
        (["CREDENTIAL"], "LOW", ""),
    ]
    for i in range(count):
        labels, risk, rationale = label_sets[i % len(label_sets)]
        examples.append({
            "text": f"Synthetic example text number {i + 1} with contextual details.",
            "labels": labels,
            "risk": risk,
            "rationale": rationale,
            "domain": domains[i % len(domains)],
        })
    return examples


def _mock_ollama_response(examples: list[dict]) -> MagicMock:
    body = json.dumps({
        "message": {"role": "assistant", "content": json.dumps(examples)},
    }).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestBuildGenerationPrompt:
    """Tests for prompt construction."""

    def test_prompt_contains_taxonomy(self):
        prompt = build_generation_prompt(count=10, domain="medical")
        for label in SpanLabel:
            assert label.value in prompt

    def test_prompt_specifies_count(self):
        prompt = build_generation_prompt(count=15, domain="medical")
        assert "15" in prompt

    def test_prompt_specifies_domain(self):
        prompt = build_generation_prompt(count=10, domain="scheduling")
        assert "scheduling" in prompt

    def test_prompt_requests_json(self):
        prompt = build_generation_prompt(count=10, domain="medical")
        assert "JSON" in prompt or "json" in prompt

    def test_prompt_forbids_real_data(self):
        prompt = build_generation_prompt(count=10, domain="medical")
        assert "synthetic" in prompt.lower() or "fictional" in prompt.lower()


class TestParseLlmResponse:
    """Tests for parsing LLM JSON output into RawExamples."""

    def test_parses_valid_response(self):
        raw = _mock_llm_response(3)
        results = parse_llm_response(raw)
        assert len(results) == 3
        assert all(isinstance(r, RawExample) for r in results)

    def test_source_is_llm_augmented(self):
        raw = _mock_llm_response(2)
        results = parse_llm_response(raw)
        assert all(r.source == "llm-augmented" for r in results)

    def test_not_hard_negative(self):
        raw = _mock_llm_response(2)
        results = parse_llm_response(raw)
        assert all(not r.is_hard_negative for r in results)

    def test_labels_are_valid_span_labels(self):
        raw = _mock_llm_response(5)
        results = parse_llm_response(raw)
        for r in results:
            for label in r.labels:
                assert isinstance(label, SpanLabel)

    def test_skips_non_dict_entries(self):
        """String entries in the response must be skipped, not crash (§3.2)."""
        raw = ["just a string", "another string"]
        results = parse_llm_response(raw)
        assert len(results) == 0

    def test_skips_mixed_string_and_dict_entries(self):
        """Mixed list of strings and dicts: keep valid dicts, skip strings."""
        valid = _mock_llm_response(2)
        raw = ["a string", valid[0], "another string", valid[1]]
        results = parse_llm_response(raw)
        assert len(results) == 2

    def test_skips_malformed_entry(self):
        raw = _mock_llm_response(3)
        raw[1] = {"text": "missing labels field"}  # malformed
        results = parse_llm_response(raw)
        assert len(results) == 2

    def test_skips_invalid_label(self):
        raw = _mock_llm_response(2)
        raw[0]["labels"] = ["NOT-A-LABEL"]
        results = parse_llm_response(raw)
        assert len(results) == 1

    def test_skips_empty_text(self):
        raw = _mock_llm_response(2)
        raw[0]["text"] = ""
        results = parse_llm_response(raw)
        assert len(results) == 1

    def test_skips_medium_single_label_no_rationale(self):
        """MEDIUM risk with 1 label and empty rationale must be rejected (tightened invariant)."""
        raw = [{
            "text": "I visit Dr. Smith every Tuesday.",
            "labels": ["ROUTINE"],
            "risk": "MEDIUM",
            "rationale": "",
            "domain": "medical",
        }]
        results = parse_llm_response(raw)
        assert len(results) == 0

    def test_accepts_medium_single_label_with_rationale(self):
        """MEDIUM risk with 1 label and non-empty rationale must be accepted."""
        raw = [{
            "text": "I visit Dr. Smith every Tuesday.",
            "labels": ["ROUTINE"],
            "risk": "MEDIUM",
            "rationale": "Weekly schedule reveals routine.",
            "domain": "medical",
        }]
        results = parse_llm_response(raw)
        assert len(results) == 1

    def test_skips_high_single_label_no_rationale(self):
        """HIGH risk with 1 label and empty rationale must be rejected."""
        raw = [{
            "text": "My government clearance is TS/SCI.",
            "labels": ["CREDENTIAL"],
            "risk": "HIGH",
            "rationale": "",
            "domain": "workplace",
        }]
        results = parse_llm_response(raw)
        assert len(results) == 0


class TestGenerateFromLlm:
    """Integration tests for generate_from_llm with mocked Ollama."""

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_returns_correct_count(self, mock_urlopen):
        mock_urlopen.return_value = _mock_ollama_response(_mock_llm_response(10))
        results = generate_from_llm(count=10, model="qwen2.5:3b")
        assert len(results) == 10

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_calls_ollama(self, mock_urlopen):
        mock_urlopen.return_value = _mock_ollama_response(_mock_llm_response(5))
        generate_from_llm(count=5, model="qwen2.5:3b")
        assert mock_urlopen.called

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_uses_specified_model(self, mock_urlopen):
        mock_urlopen.return_value = _mock_ollama_response(_mock_llm_response(5))
        generate_from_llm(count=5, model="qwen2.5:14b")
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        payload = json.loads(req.data)
        assert payload["model"] == "qwen2.5:14b"

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_all_results_are_raw_examples(self, mock_urlopen):
        mock_urlopen.return_value = _mock_ollama_response(_mock_llm_response(5))
        results = generate_from_llm(count=5, model="qwen2.5:3b")
        assert all(isinstance(r, RawExample) for r in results)

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_count_zero_raises(self, mock_urlopen):
        with pytest.raises(ValueError, match="count"):
            generate_from_llm(count=0, model="qwen2.5:3b")

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_retries_on_insufficient_results(self, mock_urlopen):
        """If first batch returns fewer than requested, makes additional calls."""
        mock_urlopen.side_effect = [
            _mock_ollama_response(_mock_llm_response(3)),
            _mock_ollama_response(_mock_llm_response(5)),
        ]
        results = generate_from_llm(count=5, model="qwen2.5:3b")
        assert len(results) == 5
        assert mock_urlopen.call_count == 2

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_connection_failure_raises(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        with pytest.raises(RuntimeError, match="Ollama"):
            generate_from_llm(count=5, model="qwen2.5:3b")

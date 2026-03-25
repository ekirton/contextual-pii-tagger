"""Tests for the output parser.

All expectations derived from specifications/output-parser.md.
"""

import json

from contextual_pii_tagger.entities import (
    DetectionResult,
    RiskLevel,
    SpanLabel,
)
from contextual_pii_tagger.output_parser import parse_output


# ── §1: Total function — never raises ──────────────────────────────────────


class TestParseOutputNeverRaises:
    """ENSURES: always returns a valid DetectionResult (never raises)."""

    def test_empty_string(self):
        result = parse_output("")
        assert isinstance(result, DetectionResult)

    def test_none_like_garbage(self):
        result = parse_output("not json at all!!!")
        assert isinstance(result, DetectionResult)

    def test_numeric_string(self):
        result = parse_output("42")
        assert isinstance(result, DetectionResult)

    def test_empty_json_object(self):
        result = parse_output("{}")
        assert isinstance(result, DetectionResult)

    def test_json_array(self):
        result = parse_output("[]")
        assert isinstance(result, DetectionResult)


# ── §2.1: JSON extraction ─────────────────────────────────────────────────


class TestJsonExtraction:
    """ENSURES: valid JSON parsed; embedded JSON extracted from code fences."""

    def test_valid_json(self):
        raw = json.dumps({
            "labels": ["WORKPLACE", "LOCATION"],
            "risk": "MEDIUM",
            "rationale": "Combined risk.",
        })
        result = parse_output(raw)
        assert SpanLabel.WORKPLACE in result.labels
        assert SpanLabel.LOCATION in result.labels

    def test_json_in_markdown_code_fence(self):
        raw = '```json\n{"labels": [], "risk": "LOW", "rationale": ""}\n```'
        result = parse_output(raw)
        assert result.risk == RiskLevel.LOW

    def test_json_with_surrounding_text(self):
        raw = 'Here is the result: {"labels": [], "risk": "LOW", "rationale": ""} end.'
        result = parse_output(raw)
        assert result.risk == RiskLevel.LOW


# ── §2.2: JSON repair ──────────────────────────────────────────────────────


class TestJsonRepair:
    """ENSURES: common malformations are repaired."""

    def test_trailing_comma(self):
        raw = '{"labels": [], "risk": "LOW", "rationale": "",}'
        result = parse_output(raw)
        assert result.risk == RiskLevel.LOW

    def test_single_quotes(self):
        raw = "{'labels': [], 'risk': 'LOW', 'rationale': ''}"
        result = parse_output(raw)
        assert result.risk == RiskLevel.LOW

    def test_unclosed_brace(self):
        raw = '{"labels": [], "risk": "LOW", "rationale": ""'
        result = parse_output(raw)
        assert isinstance(result, DetectionResult)

    def test_unquoted_keys(self):
        """Spec §2.2: unquoted keys are a repair target."""
        raw = "{labels: [], risk: 'LOW', rationale: ''}"
        result = parse_output(raw)
        assert isinstance(result, DetectionResult)


# ── §2.3: Field extraction with defaults ───────────────────────────────────


class TestFieldExtraction:
    """ENSURES: missing keys get defaults; invalid risk → LOW."""

    def test_missing_labels_defaults_to_empty(self):
        raw = json.dumps({"risk": "LOW", "rationale": ""})
        result = parse_output(raw)
        assert result.labels == frozenset()

    def test_missing_risk_defaults_to_low(self):
        raw = json.dumps({"labels": [], "rationale": ""})
        result = parse_output(raw)
        assert result.risk == RiskLevel.LOW

    def test_missing_rationale_defaults_to_empty(self):
        raw = json.dumps({"labels": [], "risk": "LOW"})
        result = parse_output(raw)
        assert result.rationale == ""

    def test_invalid_risk_value_defaults_to_low(self):
        raw = json.dumps({"labels": [], "risk": "EXTREME", "rationale": ""})
        result = parse_output(raw)
        assert result.risk == RiskLevel.LOW


# ── §2.4: Label validation ────────────────────────────────────────────────


class TestLabelValidation:
    """ENSURES: invalid labels dropped; duplicates collapsed."""

    def test_invalid_label_dropped(self):
        raw = json.dumps({
            "labels": ["INVALID_LABEL", "WORKPLACE"],
            "risk": "MEDIUM",
            "rationale": "",
        })
        result = parse_output(raw)
        assert SpanLabel.WORKPLACE in result.labels
        assert len(result.labels) == 1

    def test_all_invalid_labels_results_in_empty(self):
        raw = json.dumps({
            "labels": ["INVALID1", "INVALID2"],
            "risk": "HIGH",
            "rationale": "something",
        })
        result = parse_output(raw)
        assert result.labels == frozenset()
        assert result.risk == RiskLevel.LOW

    def test_duplicate_labels_collapsed(self):
        raw = json.dumps({
            "labels": ["LOCATION", "LOCATION", "WORKPLACE"],
            "risk": "HIGH",
            "rationale": "Combined.",
        })
        result = parse_output(raw)
        assert len(result.labels) == 2

    def test_all_valid_labels_accepted(self):
        all_labels = [label.value for label in SpanLabel]
        raw = json.dumps({
            "labels": all_labels,
            "risk": "HIGH",
            "rationale": "Everything present.",
        })
        result = parse_output(raw)
        assert len(result.labels) == 8


# ── §3: Consistency enforcement ────────────────────────────────────────────


class TestConsistencyEnforcement:
    """ENSURES: DetectionResult invariants enforced post-parse."""

    def test_empty_labels_forces_low_risk(self):
        """ENSURES: empty labels → risk LOW, rationale empty."""
        raw = json.dumps({
            "labels": [],
            "risk": "HIGH",
            "rationale": "should be cleared",
        })
        result = parse_output(raw)
        assert result.risk == RiskLevel.LOW
        assert result.rationale == ""

    def test_low_risk_clears_rationale(self):
        """ENSURES: risk LOW → rationale empty."""
        raw = json.dumps({
            "labels": ["WORKPLACE"],
            "risk": "LOW",
            "rationale": "should be cleared",
        })
        result = parse_output(raw)
        assert result.rationale == ""

    def test_high_risk_multiple_labels_missing_rationale_gets_fallback(self):
        """ENSURES: MEDIUM/HIGH + 2+ labels + empty rationale → generic fallback."""
        raw = json.dumps({
            "labels": ["WORKPLACE", "LOCATION"],
            "risk": "HIGH",
            "rationale": "",
        })
        result = parse_output(raw)
        assert result.rationale == "Multiple quasi-identifiers detected."

    def test_single_label_high_risk_empty_rationale_stays_empty(self):
        """ENSURES: MEDIUM/HIGH + 1 label + empty rationale → remains empty (no fallback)."""
        raw = json.dumps({
            "labels": ["WORKPLACE"],
            "risk": "HIGH",
            "rationale": "",
        })
        result = parse_output(raw)
        assert result.risk == RiskLevel.HIGH
        assert result.rationale == ""


# ── §2.5: Fallback ────────────────────────────────────────────────────────


class TestFallback:
    """ENSURES: total failure → empty DetectionResult."""

    def test_completely_unparseable(self):
        result = parse_output("{{{{garbage!@#$")
        assert result.labels == frozenset()
        assert result.risk == RiskLevel.LOW
        assert result.rationale == ""

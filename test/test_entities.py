"""Tests for core entities: SpanLabel, RiskLevel, DetectionResult.

All expectations derived from specifications/entities.md §1-3.
"""

import pytest

from contextual_pii_tagger.entities import (
    DetectionResult,
    RiskLevel,
    SpanLabel,
)


# ── SpanLabel (entities.md §1) ─────────────────────────────────────────────


class TestSpanLabel:
    """MAINTAINS: 8 fixed values, uppercase with hyphens, no Tier 1 overlap."""

    def test_has_exactly_eight_values(self):
        assert len(SpanLabel) == 8

    def test_all_values_present(self):
        expected = {
            "LOCATION",
            "WORKPLACE",
            "ROUTINE",
            "MEDICAL-CONTEXT",
            "DEMOGRAPHIC",
            "DEVICE-ID",
            "CREDENTIAL",
            "QUASI-ID",
        }
        actual = {label.value for label in SpanLabel}
        assert actual == expected

    def test_values_are_uppercase_strings(self):
        for label in SpanLabel:
            assert label.value == label.value.upper()
            assert isinstance(label.value, str)

    def test_no_overlap_with_tier1_labels(self):
        tier1 = {"NAME", "EMAIL", "PHONE", "ADDRESS", "GOV-ID", "FINANCIAL", "DOB", "BIOMETRIC"}
        tier2 = {label.value for label in SpanLabel}
        assert tier1.isdisjoint(tier2)

    def test_constructible_from_string(self):
        assert SpanLabel("LOCATION") == SpanLabel.LOCATION
        assert SpanLabel("QUASI-ID") == SpanLabel.QUASI_ID

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            SpanLabel("INVALID")


# ── RiskLevel (entities.md §2) ─────────────────────────────────────────────


class TestRiskLevel:
    """MAINTAINS: 3 values, ordered LOW < MEDIUM < HIGH."""

    def test_has_exactly_three_values(self):
        assert len(RiskLevel) == 3

    def test_all_values_present(self):
        expected = {"LOW", "MEDIUM", "HIGH"}
        actual = {level.value for level in RiskLevel}
        assert actual == expected

    def test_values_are_uppercase_no_separators(self):
        for level in RiskLevel:
            assert level.value == level.value.upper()
            assert "-" not in level.value

    def test_constructible_from_string(self):
        assert RiskLevel("LOW") == RiskLevel.LOW
        assert RiskLevel("HIGH") == RiskLevel.HIGH

    def test_ordering_low_medium_high(self):
        """MAINTAINS: ordered by severity LOW < MEDIUM < HIGH."""
        values = list(RiskLevel)
        assert values.index(RiskLevel.LOW) < values.index(RiskLevel.MEDIUM)
        assert values.index(RiskLevel.MEDIUM) < values.index(RiskLevel.HIGH)


# ── DetectionResult (entities.md §3) ───────────────────────────────────────


class TestDetectionResult:
    """REQUIRES/ENSURES/MAINTAINS on construction and invariants."""

    def test_empty_labels_requires_low_risk(self):
        """ENSURES: empty labels → risk LOW."""
        result = DetectionResult(labels=frozenset(), risk=RiskLevel.LOW, rationale="")
        assert result.risk == RiskLevel.LOW
        assert result.labels == frozenset()

    def test_empty_labels_with_non_low_risk_raises(self):
        """ENSURES: empty labels → risk must be LOW."""
        with pytest.raises(ValueError):
            DetectionResult(labels=frozenset(), risk=RiskLevel.HIGH, rationale="something")

    def test_low_risk_requires_empty_rationale(self):
        """ENSURES: risk LOW → rationale empty."""
        with pytest.raises(ValueError):
            DetectionResult(
                labels=frozenset({SpanLabel.LOCATION}),
                risk=RiskLevel.LOW,
                rationale="should not be here",
            )

    def test_medium_risk_with_multiple_labels_requires_rationale(self):
        """ENSURES: MEDIUM/HIGH with 2+ labels → rationale non-empty."""
        with pytest.raises(ValueError):
            DetectionResult(
                labels=frozenset({SpanLabel.LOCATION, SpanLabel.WORKPLACE}),
                risk=RiskLevel.MEDIUM,
                rationale="",
            )

    def test_high_risk_with_multiple_labels_and_rationale_valid(self):
        result = DetectionResult(
            labels=frozenset({SpanLabel.LOCATION, SpanLabel.WORKPLACE}),
            risk=RiskLevel.HIGH,
            rationale="Combination identifies individual.",
        )
        assert result.risk == RiskLevel.HIGH
        assert len(result.labels) == 2

    def test_single_label_medium_risk_empty_rationale_allowed(self):
        """ENSURES: rationale required only when 2+ labels at MEDIUM/HIGH."""
        result = DetectionResult(
            labels=frozenset({SpanLabel.LOCATION}),
            risk=RiskLevel.MEDIUM,
            rationale="",
        )
        assert result.risk == RiskLevel.MEDIUM

    def test_immutable(self):
        """MAINTAINS: immutable after construction."""
        result = DetectionResult(labels=frozenset(), risk=RiskLevel.LOW, rationale="")
        with pytest.raises(AttributeError):
            result.risk = RiskLevel.HIGH

    def test_to_dict_round_trip(self):
        """Serialization: to_dict/from_dict round-trip without data loss."""
        original = DetectionResult(
            labels=frozenset({SpanLabel.LOCATION, SpanLabel.WORKPLACE}),
            risk=RiskLevel.HIGH,
            rationale="Combined risk.",
        )
        d = original.to_dict()
        restored = DetectionResult.from_dict(d)
        assert restored.risk == original.risk
        assert restored.rationale == original.rationale
        assert restored.labels == original.labels

    def test_to_dict_empty_result(self):
        result = DetectionResult(labels=frozenset(), risk=RiskLevel.LOW, rationale="")
        d = result.to_dict()
        assert d["labels"] == []
        assert d["risk"] == "LOW"
        assert d["rationale"] == ""
        restored = DetectionResult.from_dict(d)
        assert restored.labels == frozenset()

    def test_single_label_high_risk_empty_rationale_allowed(self):
        """ENSURES: rationale required only when 2+ labels. Single label HIGH is fine."""
        result = DetectionResult(
            labels=frozenset({SpanLabel.CREDENTIAL}),
            risk=RiskLevel.HIGH,
            rationale="",
        )
        assert result.risk == RiskLevel.HIGH
        assert result.rationale == ""

    def test_to_dict_labels_are_sorted(self):
        """Labels in serialized output should be sorted for determinism."""
        result = DetectionResult(
            labels=frozenset({SpanLabel.WORKPLACE, SpanLabel.LOCATION}),
            risk=RiskLevel.HIGH,
            rationale="Risk.",
        )
        d = result.to_dict()
        assert d["labels"] == sorted(d["labels"])

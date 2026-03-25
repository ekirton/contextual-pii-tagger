"""Tests for RawExample entity.

All expectations derived from specifications/data-generation.md §2.0.
"""

import pytest

from contextual_pii_tagger.entities import (
    RiskLevel,
    SpanLabel,
)
from contextual_pii_tagger.data.raw_example import RawExample


class TestRawExample:
    """REQUIRES/ENSURES on construction; DetectionResult invariants."""

    def test_valid_construction(self):
        ex = RawExample(
            text="I saw my cardiologist at City Hospital last Monday",
            labels=frozenset({SpanLabel.MEDICAL_CONTEXT, SpanLabel.WORKPLACE, SpanLabel.ROUTINE}),
            risk=RiskLevel.MEDIUM,
            rationale="Medical specialty and specific hospital narrow identification.",
            is_hard_negative=False,
            domain="medical",
            source="template",
        )
        assert ex.text.startswith("I saw")
        assert ex.domain == "medical"
        assert ex.source == "template"

    def test_empty_text_raises(self):
        """REQUIRES: text non-empty."""
        with pytest.raises(ValueError):
            RawExample(
                text="",
                labels=frozenset({SpanLabel.LOCATION}),
                risk=RiskLevel.MEDIUM,
                rationale="",
                is_hard_negative=False,
                domain="medical",
                source="template",
            )

    def test_invalid_domain_raises(self):
        """REQUIRES: domain in VALID_DOMAINS."""
        with pytest.raises(ValueError):
            RawExample(
                text="some text",
                labels=frozenset(),
                risk=RiskLevel.LOW,
                rationale="",
                is_hard_negative=False,
                domain="invalid",
                source="template",
            )

    def test_invalid_source_raises(self):
        """REQUIRES: source in VALID_SOURCES."""
        with pytest.raises(ValueError):
            RawExample(
                text="some text",
                labels=frozenset(),
                risk=RiskLevel.LOW,
                rationale="",
                is_hard_negative=False,
                domain="medical",
                source="unknown",
            )

    def test_detection_result_invariants(self):
        """Empty labels→LOW; LOW→empty rationale; MEDIUM/HIGH+2 labels→non-empty rationale."""
        # Empty labels must be LOW
        with pytest.raises(ValueError):
            RawExample(
                text="some text",
                labels=frozenset(),
                risk=RiskLevel.MEDIUM,
                rationale="reason",
                is_hard_negative=False,
                domain="medical",
                source="template",
            )

        # LOW risk must have empty rationale
        with pytest.raises(ValueError):
            RawExample(
                text="some text",
                labels=frozenset({SpanLabel.LOCATION}),
                risk=RiskLevel.LOW,
                rationale="should be empty",
                is_hard_negative=False,
                domain="medical",
                source="template",
            )

        # MEDIUM/HIGH with 2+ labels requires non-empty rationale
        with pytest.raises(ValueError):
            RawExample(
                text="some text",
                labels=frozenset({SpanLabel.LOCATION, SpanLabel.ROUTINE}),
                risk=RiskLevel.MEDIUM,
                rationale="",
                is_hard_negative=False,
                domain="medical",
                source="template",
            )

    def test_frozen(self):
        """MAINTAINS: immutable after construction."""
        ex = RawExample(
            text="some text",
            labels=frozenset({SpanLabel.LOCATION}),
            risk=RiskLevel.MEDIUM,
            rationale="",
            is_hard_negative=False,
            domain="medical",
            source="template",
        )
        with pytest.raises(AttributeError):
            ex.text = "changed"  # type: ignore[misc]

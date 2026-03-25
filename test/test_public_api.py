"""Tests for the package public API (F-04).

Verifies that PIIDetector is importable from the top-level package,
alongside the existing entity exports.

Spec: specifications/detection-interface.md §1
"""

from contextual_pii_tagger import (
    DetectionResult,
    PIIDetector,
    RiskLevel,
    SpanLabel,
)


class TestPublicAPI:
    """ENSURES: PIIDetector is the primary public entry point."""

    def test_pii_detector_importable_from_top_level(self):
        """The top-level import exposes PIIDetector."""
        assert PIIDetector is not None

    def test_pii_detector_has_from_pretrained(self):
        """PIIDetector exposes the from_pretrained factory method."""
        assert hasattr(PIIDetector, "from_pretrained")
        assert callable(PIIDetector.from_pretrained)

    def test_pii_detector_has_detect(self):
        """PIIDetector exposes the detect method."""
        assert hasattr(PIIDetector, "detect")
        assert callable(PIIDetector.detect)

    def test_all_exports_listed(self):
        """__all__ includes PIIDetector alongside entity types."""
        import contextual_pii_tagger

        assert "PIIDetector" in contextual_pii_tagger.__all__
        assert "DetectionResult" in contextual_pii_tagger.__all__
        assert "RiskLevel" in contextual_pii_tagger.__all__
        assert "SpanLabel" in contextual_pii_tagger.__all__

"""Tests for PIIDetector.

All expectations derived from specifications/detection-interface.md §1, §3.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from contextual_pii_tagger.entities import (
    DetectionResult,
    RiskLevel,
    SpanLabel,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _mock_model():
    """Create a mock model that returns valid JSON output."""
    model = MagicMock()
    model.device = "cpu"
    return model


def _mock_tokenizer():
    """Create a mock tokenizer with encode/decode."""
    tok = MagicMock()
    tok.encode = MagicMock(side_effect=lambda text, **kw: list(range(len(text.split()))))
    tok.decode = MagicMock(side_effect=lambda ids, **kw: " ".join(f"w{i}" for i in ids))
    tok.eos_token_id = 2
    tok.pad_token_id = 0
    return tok


def _valid_json_output(labels=None, risk="LOW", rationale=""):
    return json.dumps({
        "labels": labels or [],
        "risk": risk,
        "rationale": rationale,
    })


# ── §1.1: from_pretrained ───────────────────────────────────────────────


class TestFromPretrained:
    """ENSURES: model loaded and ready; RAISES on invalid path."""

    @patch("contextual_pii_tagger.detector.AutoTokenizer")
    @patch("contextual_pii_tagger.detector.AutoModelForCausalLM")
    def test_loads_from_local_path(self, mock_model_cls, mock_tok_cls, tmp_path):
        from contextual_pii_tagger.detector import PIIDetector

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        mock_model_cls.from_pretrained.return_value = _mock_model()

        detector = PIIDetector.from_pretrained(str(model_dir))
        assert detector is not None
        mock_tok_cls.from_pretrained.assert_called_once()
        mock_model_cls.from_pretrained.assert_called_once()

    def test_raises_file_not_found_for_missing_local_path(self):
        from contextual_pii_tagger.detector import PIIDetector

        with pytest.raises(FileNotFoundError):
            PIIDetector.from_pretrained("/nonexistent/path/model")

    @patch("contextual_pii_tagger.detector.AutoTokenizer")
    @patch("contextual_pii_tagger.detector.AutoModelForCausalLM")
    def test_loads_adapter_when_adapter_config_present(
        self, mock_model_cls, mock_tok_cls, tmp_path
    ):
        from contextual_pii_tagger.detector import PIIDetector

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "adapter_config.json").write_text("{}")

        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        mock_model = _mock_model()
        mock_model_cls.from_pretrained.return_value = mock_model

        with patch("contextual_pii_tagger.detector.PeftModel") as mock_peft:
            mock_peft.from_pretrained.return_value = mock_model
            detector = PIIDetector.from_pretrained(str(model_dir))
            assert detector is not None


# ── §1.2: detect ────────────────────────────────────────────────────────


class TestDetect:
    """ENSURES: returns valid DetectionResult; RAISES on empty text."""

    def _make_detector(self):
        from contextual_pii_tagger.detector import PIIDetector

        detector = PIIDetector.__new__(PIIDetector)
        detector._model = _mock_model()
        detector._tokenizer = _mock_tokenizer()
        return detector

    @patch("contextual_pii_tagger.detector._generate")
    def test_returns_detection_result(self, mock_gen):
        mock_gen.return_value = _valid_json_output()
        detector = self._make_detector()
        result = detector.detect("Some text with PII.")
        assert isinstance(result, DetectionResult)

    def test_raises_on_empty_text(self):
        detector = self._make_detector()
        with pytest.raises(ValueError):
            detector.detect("")

    @patch("contextual_pii_tagger.detector._generate")
    def test_result_satisfies_invariants_labels_present(self, mock_gen):
        mock_gen.return_value = _valid_json_output(
            labels=["LOCATION", "WORKPLACE"],
            risk="HIGH",
            rationale="Combined risk of re-identification.",
        )
        detector = self._make_detector()
        result = detector.detect("I work at Acme near downtown.")
        assert SpanLabel.LOCATION in result.labels
        assert SpanLabel.WORKPLACE in result.labels
        assert result.risk == RiskLevel.HIGH

    @patch("contextual_pii_tagger.detector._generate")
    def test_result_satisfies_invariants_clean_text(self, mock_gen):
        mock_gen.return_value = _valid_json_output()
        detector = self._make_detector()
        result = detector.detect("The sky is blue.")
        assert result.labels == frozenset()
        assert result.risk == RiskLevel.LOW
        assert result.rationale == ""

    @patch("contextual_pii_tagger.detector._generate")
    def test_deterministic(self, mock_gen):
        """ENSURES: same input produces same output."""
        mock_gen.return_value = _valid_json_output(
            labels=["LOCATION"], risk="MEDIUM", rationale=""
        )
        detector = self._make_detector()
        r1 = detector.detect("Near the park.")
        r2 = detector.detect("Near the park.")
        assert r1 == r2

    @patch("contextual_pii_tagger.detector._generate")
    def test_malformed_output_returns_fallback(self, mock_gen):
        """ENSURES: output parser handles malformed model output gracefully."""
        mock_gen.return_value = "not valid json at all"
        detector = self._make_detector()
        result = detector.detect("Some text.")
        assert isinstance(result, DetectionResult)
        assert result.labels == frozenset()
        assert result.risk == RiskLevel.LOW

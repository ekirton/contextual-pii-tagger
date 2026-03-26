"""Tests for XGBoost baseline.

All expectations derived from specifications/evaluation.md §8.
"""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from contextual_pii_tagger.entities import (
    DetectionResult,
    RiskLevel,
    SpanLabel,
)
from contextual_pii_tagger.example import Example


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_example(
    text="I work at Acme Corp near downtown.",
    labels=None,
    risk=RiskLevel.MEDIUM,
    rationale="Combined risk.",
    is_hard_negative=False,
    idx=1,
):
    if labels is None:
        labels = frozenset({SpanLabel.WORKPLACE, SpanLabel.LOCATION})
    return Example(
        id=f"train-{idx:05d}",
        text=text,
        labels=labels,
        risk=risk,
        rationale=rationale,
        is_hard_negative=is_hard_negative,
        split="train",
        domain="workplace",
        source="hard-negative" if is_hard_negative else "template",
    )


def _make_clean_example(idx=2):
    return _make_example(
        text="The weather is nice today.",
        labels=frozenset(),
        risk=RiskLevel.LOW,
        rationale="",
        is_hard_negative=True,
        idx=idx,
    )


@contextmanager
def _mock_optional_deps():
    """Mock xgboost and spacy if not installed."""
    mocked = {}
    saved = {}

    # xgboost
    try:
        import xgboost  # noqa: F401
    except ImportError:
        import numpy as _np

        xgb_mod = ModuleType("xgboost")
        xgb_mod.__spec__ = MagicMock()

        class _FakeXGBClassifier:
            def __init__(self, **kwargs):
                self._classes = None

            def fit(self, X, y):
                self._classes = sorted(set(y))

            def predict_proba(self, X):
                n = X.shape[0]
                nc = max(len(self._classes) if self._classes else 2, 2)
                return _np.full((n, nc), 1.0 / nc)

            def predict(self, X):
                return _np.zeros(X.shape[0], dtype=int)

        xgb_mod.XGBClassifier = _FakeXGBClassifier
        saved["xgboost"] = sys.modules.get("xgboost")
        sys.modules["xgboost"] = xgb_mod
        mocked["xgboost"] = xgb_mod

    # spacy
    try:
        import spacy  # noqa: F401
    except ImportError:
        spacy_mod = ModuleType("spacy")
        spacy_mod.__spec__ = MagicMock()
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_doc.sents = [MagicMock()]
        mock_nlp.return_value = mock_doc
        spacy_mod.load = MagicMock(return_value=mock_nlp)
        saved["spacy"] = sys.modules.get("spacy")
        sys.modules["spacy"] = spacy_mod
        mocked["spacy"] = spacy_mod

    # Clear cached module
    mod_name = "contextual_pii_tagger.eval.baseline"
    saved_mod = sys.modules.pop(mod_name, None)

    try:
        yield importlib.import_module(mod_name)
    finally:
        if saved_mod is not None:
            sys.modules[mod_name] = saved_mod
        else:
            sys.modules.pop(mod_name, None)
        for name, orig in saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


# ── §8.2: extract_features ──────────────────────────────────────────────


class TestExtractFeatures:
    """ENSURES: deterministic fixed-dimension feature vector."""

    def test_returns_dict(self):
        with _mock_optional_deps() as mod:
            feats = mod.extract_features("I work at Acme Corp near downtown.")
            assert isinstance(feats, dict)

    def test_contains_text_statistics(self):
        with _mock_optional_deps() as mod:
            feats = mod.extract_features("I work at Acme Corp.")
            assert "token_count" in feats
            assert "sentence_count" in feats
            assert "avg_sentence_length" in feats

    def test_contains_pattern_indicators(self):
        with _mock_optional_deps() as mod:
            feats = mod.extract_features("Email me at test@example.com")
            assert "has_email_pattern" in feats

    def test_deterministic(self):
        with _mock_optional_deps() as mod:
            text = "I work at Acme Corp near downtown."
            f1 = mod.extract_features(text)
            f2 = mod.extract_features(text)
            assert f1 == f2


# ── §8.1: train_baseline / XGBoostPredictor ─────────────────────────────


class TestTrainBaseline:
    """ENSURES: returns predictor implementing predict(text) -> DetectionResult."""

    def test_returns_predictor(self):
        train_data = [_make_example(idx=i) for i in range(1, 6)] + [
            _make_clean_example(idx=i) for i in range(6, 11)
        ]
        with _mock_optional_deps() as mod:
            predictor = mod.train_baseline(train_data)
            assert hasattr(predictor, "predict")

    def test_predict_returns_detection_result(self):
        train_data = [_make_example(idx=i) for i in range(1, 6)] + [
            _make_clean_example(idx=i) for i in range(6, 11)
        ]
        with _mock_optional_deps() as mod:
            predictor = mod.train_baseline(train_data)
            result = predictor.predict("Some text to classify.")
            assert isinstance(result, DetectionResult)

    def test_rationale_nonempty_when_risk_medium_or_high(self):
        """ENSURES: rationale is non-empty when risk is MEDIUM or HIGH (entities.md §3)."""
        train_data = [_make_example(idx=i) for i in range(1, 6)] + [
            _make_clean_example(idx=i) for i in range(6, 11)
        ]
        with _mock_optional_deps() as mod:
            predictor = mod.train_baseline(train_data)
            result = predictor.predict("I work at Acme Corp near downtown.")
            if result.risk in (RiskLevel.MEDIUM, RiskLevel.HIGH):
                assert result.rationale != ""
            else:
                assert result.rationale == ""

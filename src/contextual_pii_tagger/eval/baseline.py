"""XGBoost baseline: non-contextual classifier for comparison.

Spec: specifications/evaluation.md §8

Heavy dependencies (xgboost, spacy) imported lazily so the module can
be imported and tested without them installed.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from contextual_pii_tagger.entities import (
    DetectionResult,
    RiskLevel,
    SpanLabel,
)

if TYPE_CHECKING:
    from contextual_pii_tagger.example import Example

logger = logging.getLogger(__name__)

# Ordered list for consistent indexing
_LABELS = sorted(SpanLabel, key=lambda s: s.value)
_RISK_LEVELS = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]

# ── Pattern indicators ───────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
_ADDRESS_RE = re.compile(r"\b\d+\s+[A-Z][a-z]+\s+(?:St|Ave|Rd|Blvd|Dr|Ln)\b")
_CRED_RE = re.compile(r"\b(?:sk-[A-Za-z0-9]+|api[_-]?key|password|token)\b", re.IGNORECASE)


def extract_features(text: str) -> dict:
    """Compute a feature dict from *text*.

    ENSURES:
        - Deterministic output.
        - Contains text statistics, pattern indicators, and entity counts.
    """
    words = text.split()
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    features: dict = {
        # Text statistics
        "token_count": len(words),
        "sentence_count": len(sentences) if sentences else 1,
        "avg_sentence_length": (
            len(words) / len(sentences) if sentences else float(len(words))
        ),
        # Pattern indicators
        "has_email_pattern": int(bool(_EMAIL_RE.search(text))),
        "has_phone_pattern": int(bool(_PHONE_RE.search(text))),
        "has_address_pattern": int(bool(_ADDRESS_RE.search(text))),
        "has_credential_pattern": int(bool(_CRED_RE.search(text))),
    }

    # Entity counts from spaCy (if available)
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        ent_counts = {"PERSON": 0, "ORG": 0, "GPE": 0, "DATE": 0}
        for ent in doc.ents:
            if ent.label_ in ent_counts:
                ent_counts[ent.label_] += 1
        features.update({f"ent_{k}": v for k, v in ent_counts.items()})
    except (ImportError, OSError):
        features.update({
            "ent_PERSON": 0,
            "ent_ORG": 0,
            "ent_GPE": 0,
            "ent_DATE": 0,
        })

    return features


class XGBoostPredictor:
    """Wrapper that implements predict(text) -> DetectionResult.

    ENSURES: rationale is always empty string.
    """

    def __init__(
        self,
        label_classifiers: dict[SpanLabel, object],
        risk_classifier: object,
        tfidf: TfidfVectorizer,
    ) -> None:
        self._label_clfs = label_classifiers
        self._risk_clf = risk_classifier
        self._tfidf = tfidf

    def save(self, path: str | Path) -> None:
        """Persist the predictor to *path*."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "xgboost_baseline.pkl", "wb") as f:
            pickle.dump(
                {
                    "label_classifiers": self._label_clfs,
                    "risk_classifier": self._risk_clf,
                    "tfidf": self._tfidf,
                },
                f,
            )
        logger.info("XGBoost baseline saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> XGBoostPredictor:
        """Load a predictor from *path*."""
        path = Path(path)
        with open(path / "xgboost_baseline.pkl", "rb") as f:
            data = pickle.load(f)  # noqa: S301
        logger.info("XGBoost baseline loaded from %s", path)
        return cls(data["label_classifiers"], data["risk_classifier"], data["tfidf"])

    def predict(self, text: str) -> DetectionResult:
        """Classify *text* and return a DetectionResult."""
        feats = extract_features(text)
        feat_values = [feats[k] for k in sorted(feats.keys())]

        # TF-IDF features
        tfidf_vec = self._tfidf.transform([text]).toarray()[0]
        full_features = np.concatenate([tfidf_vec, feat_values])
        X = full_features.reshape(1, -1)

        # Multilabel classification
        labels: set[SpanLabel] = set()
        for label in _LABELS:
            clf = self._label_clfs[label]
            prob = clf.predict_proba(X)[0]
            # prob shape depends on classes_ — take positive class if binary
            if len(prob) > 1:
                pos_prob = prob[1]
            else:
                pos_prob = prob[0]
            if pos_prob >= 0.5:
                labels.add(label)

        # Risk classification
        risk_pred_idx = self._risk_clf.predict(X)[0]
        risk = _RISK_LEVELS[int(risk_pred_idx)]

        # Enforce invariants
        if not labels:
            risk = RiskLevel.LOW

        # DetectionResult requires rationale when risk >= MEDIUM and 2+ labels
        rationale = ""
        if risk in (RiskLevel.MEDIUM, RiskLevel.HIGH) and len(labels) >= 2:
            sorted_labels = sorted(l.value for l in labels)
            rationale = f"Multiple PII types detected: {', '.join(sorted_labels)}"

        return DetectionResult(
            labels=frozenset(labels),
            risk=risk,
            rationale=rationale,
        )


def train_baseline(train_dataset: list[Example]) -> XGBoostPredictor:
    """Train XGBoost classifiers on the training split.

    REQUIRES:
        - *train_dataset* is the training split.

    ENSURES:
        - Returns an XGBoostPredictor implementing predict(text) -> DetectionResult.
        - No use of validation or test data.
    """
    from xgboost import XGBClassifier

    texts = [ex.text for ex in train_dataset]

    # Fit TF-IDF on training vocabulary
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(texts).toarray()

    # Build feature matrix
    feat_dicts = [extract_features(ex.text) for ex in train_dataset]
    feat_keys = sorted(feat_dicts[0].keys())
    feat_matrix = np.array([[d[k] for k in feat_keys] for d in feat_dicts])
    X = np.hstack([tfidf_matrix, feat_matrix])

    # Train one binary classifier per SpanLabel
    label_clfs: dict[SpanLabel, object] = {}
    for label in _LABELS:
        y = np.array([1 if label in ex.labels else 0 for ex in train_dataset])
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        # If all labels are the same class, train a dummy
        if len(set(y)) < 2:
            clf = _DummyClassifier(y[0])
        else:
            clf.fit(X, y)
        label_clfs[label] = clf

    # Train risk classifier (multiclass)
    risk_map = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
    y_risk = np.array([risk_map[ex.risk] for ex in train_dataset])
    risk_clf = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        objective="multi:softmax",
        num_class=3,
        verbosity=0,
    )
    if len(set(y_risk)) < 2:
        risk_clf = _DummyClassifier(y_risk[0], num_classes=3)
    else:
        risk_clf.fit(X, y_risk)

    logger.info("XGBoost baseline trained on %d examples", len(train_dataset))
    return XGBoostPredictor(label_clfs, risk_clf, tfidf)


class _DummyClassifier:
    """Fallback when all training labels are the same class."""

    def __init__(self, constant: int, num_classes: int = 2) -> None:
        self._constant = constant
        self._num_classes = num_classes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        probs = np.zeros((n, self._num_classes))
        probs[:, int(self._constant)] = 1.0
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._constant)

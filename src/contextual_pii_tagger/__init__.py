"""contextual-pii-tagger: Contextual PII detection via QLoRA fine-tuned SLM."""

from contextual_pii_tagger.entities import (
    DetectionResult,
    RiskLevel,
    SpanLabel,
)

__all__ = [
    "DetectionResult",
    "PIIDetector",
    "RiskLevel",
    "SpanLabel",
]


def __getattr__(name: str):
    if name == "PIIDetector":
        from contextual_pii_tagger.detector import PIIDetector

        return PIIDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

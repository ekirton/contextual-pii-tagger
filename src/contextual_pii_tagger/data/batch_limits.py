"""Model-aware batch size limits for LLM generation stages.

Each model has a max output token budget. We derive per-call example limits
from that budget and the estimated tokens per example for each output format.

Local models (Ollama) typically have ~4K usable output tokens. Larger
context windows exist but generation quality degrades with long outputs.

We reserve ~20 % headroom for JSON framing and variance.
"""

from __future__ import annotations

# Usable output tokens after reserving headroom.
_MODEL_TOKEN_BUDGET: dict[str, int] = {
    "qwen2.5:3b": 2_400,
    "qwen2.5:7b": 3_200,
    "qwen2.5:14b": 3_200,
    "llama3.1:8b": 3_200,
    "mistral:7b": 3_200,
}

_DEFAULT_BUDGET = 3_200  # fallback: assume local 7B model

# Estimated tokens per example by output format.
_TOKENS_PER_STRUCTURED_EXAMPLE = 75  # Stage 2: JSON object with text/labels/risk/…
_TOKENS_PER_SIMPLE_EXAMPLE = 30  # Stage 4: plain string (1-2 sentences)
_TOKENS_PER_VALIDATION_EXAMPLE = 80  # Stage 3: input + corrected output per example


def max_batch_structured(model: str) -> int:
    """Max examples per call for structured JSON generation (Stage 2)."""
    budget = _MODEL_TOKEN_BUDGET.get(model, _DEFAULT_BUDGET)
    return max(10, budget // _TOKENS_PER_STRUCTURED_EXAMPLE)


def max_batch_simple(model: str) -> int:
    """Max examples per call for simple string generation (Stage 4)."""
    budget = _MODEL_TOKEN_BUDGET.get(model, _DEFAULT_BUDGET)
    return max(10, budget // _TOKENS_PER_SIMPLE_EXAMPLE)


def max_batch_validation(model: str) -> int:
    """Max examples per call for validation (Stage 3)."""
    budget = _MODEL_TOKEN_BUDGET.get(model, _DEFAULT_BUDGET)
    return max(10, budget // _TOKENS_PER_VALIDATION_EXAMPLE)

"""Training data formatting: Example → prompt-completion pair.

Spec: specifications/training.md §1-2
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from contextual_pii_tagger.example import Example
from contextual_pii_tagger.prompt import MAX_SEQUENCE_LENGTH, get_template_text

logger = logging.getLogger(__name__)


def format_example(example: Example, tokenizer: object) -> dict[str, str] | None:
    """Convert an *Example* into a formatted training string.

    Returns a dict with a ``"text"`` key containing the full
    prompt + completion, or ``None`` if the example exceeds the token
    budget (1,024 tokens).
    """
    # Build prompt portion
    prompt = get_template_text(example.text)

    # Build completion portion — compact JSON
    completion_obj = {
        "labels": sorted(label.value for label in example.labels),
        "risk": example.risk.value,
        "rationale": example.rationale,
    }
    completion_json = json.dumps(completion_obj, separators=(",", ":"))
    completion = f"{completion_json}\n<|end|>"

    full_text = prompt + completion

    # Check token count
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    if len(tokens) > MAX_SEQUENCE_LENGTH:
        logger.warning(
            "Skipping example %s: %d tokens exceeds %d limit",
            example.id,
            len(tokens),
            MAX_SEQUENCE_LENGTH,
        )
        return None

    return {"text": full_text}


def prepare_dataset(
    dataset_path: str, tokenizer: object
) -> list[dict[str, str]]:
    """Load train.jsonl, format all examples, exclude oversize, shuffle.

    Returns a list of dicts with ``"text"`` keys, suitable for SFTTrainer.
    """
    train_file = Path(dataset_path) / "train.jsonl"
    examples: list[Example] = []
    with open(train_file) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(Example.from_dict(json.loads(line)))

    formatted: list[dict[str, str]] = []
    skipped = 0
    for ex in examples:
        result = format_example(ex, tokenizer)
        if result is not None:
            formatted.append(result)
        else:
            skipped += 1

    if skipped > 0:
        logger.info("Skipped %d/%d examples exceeding token limit", skipped, len(examples))

    random.shuffle(formatted)
    return formatted

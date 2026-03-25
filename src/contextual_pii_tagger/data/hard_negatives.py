"""Hard negative injection (Stage 4) via Ollama.

Spec: specifications/data-generation.md §5
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from contextual_pii_tagger.entities import RiskLevel
from contextual_pii_tagger.example import Example

logger = logging.getLogger(__name__)

from contextual_pii_tagger.data.batch_limits import max_batch_simple
from contextual_pii_tagger.data.cli_utils import call_ollama

_BATCH_MULTIPLIER = 1.3
_MAX_RETRIES = 5


def compute_hard_negative_counts(
    split_counts: dict[str, int],
    ratio: float,
) -> dict[str, int]:
    """Compute how many hard negatives each split needs.

    For each split: hn_count / (existing_count + hn_count) = ratio
    So: hn_count = existing_count * ratio / (1 - ratio)
    """
    if ratio <= 0.0:
        return {s: 0 for s in split_counts}

    result: dict[str, int] = {}
    for split, existing in split_counts.items():
        if existing == 0:
            result[split] = 0
        else:
            result[split] = round(existing * ratio / (1 - ratio))
    return result


def build_hard_negative_prompt(count: int) -> str:
    """Build prompt for generating hard negative texts."""
    return f"""\
Generate exactly {count} short text examples (1-2 sentences each) that mention \
locations, times, organizations, or people but are NOT personally identifiable \
information (PII).

These are "hard negatives" — they look like PII at first glance but are not. \
Categories include:

- Historical references: "Abraham Lincoln delivered the Gettysburg Address in 1863."
- Fictional characters: "Sherlock Holmes lived at 221B Baker Street."
- Public figures in public contexts: "The president spoke at the White House today."
- Generic statements: "Many people commute by train in large cities."
- Hypothetical scenarios: "If someone lived near a hospital, they might walk to appointments."

Requirements:
- Each text must be entirely fictional or reference public knowledge
- Each text must NOT contain real personal information about private individuals
- Include a mix of categories above
- Texts should sound natural and varied

Return a JSON object with a single key "texts" containing an array of strings.

Example output for 2 examples:
{{"texts": ["Abraham Lincoln delivered the Gettysburg Address in 1863.", "Many hospitals in the Portland area offer pediatric care."]}}

Each element in the "texts" array MUST be a string."""


def parse_hard_negative_response(texts: list[Any]) -> list[str]:
    """Parse the list of hard negative texts, filtering empties."""
    return [str(t) for t in texts if t and str(t).strip()]


def inject_hard_negatives(
    examples: list[Example],
    ratio: float,
    model: str = "sonnet",
) -> list[Example]:
    """Inject hard negative examples into each split.

    Spec: specifications/data-generation.md §5.1
    """
    if ratio <= 0.0:
        return list(examples)

    # Count existing examples per split
    split_counts: dict[str, int] = {}
    split_max_num: dict[str, int] = {}
    for ex in examples:
        split_counts[ex.split] = split_counts.get(ex.split, 0) + 1
        num = int(ex.id.split("-", 1)[1])
        split_max_num[ex.split] = max(split_max_num.get(ex.split, 0), num)

    hn_counts = compute_hard_negative_counts(split_counts, ratio)
    total_needed = sum(hn_counts.values())

    if total_needed == 0:
        return list(examples)

    # Generate hard negative texts in batches
    request_count = int(math.ceil(total_needed * _BATCH_MULTIPLIER))
    batch_cap = max_batch_simple(model)
    total_batches = (request_count + batch_cap - 1) // batch_cap
    print(
        f"[Stage 4] Generating {total_needed} hard negatives "
        f"(requesting {request_count}, batch cap: {batch_cap}/call, model: {model})"
    )
    texts: list[str] = []
    consecutive_failures = 0
    batch_num = 0

    while len(texts) < request_count and consecutive_failures < _MAX_RETRIES:
        batch_size = min(batch_cap, request_count - len(texts))
        batch_num += 1
        prompt = build_hard_negative_prompt(count=batch_size)
        try:
            raw_texts = call_ollama(prompt, model)
            parsed = parse_hard_negative_response(raw_texts)
            if parsed:
                texts.extend(parsed)
                consecutive_failures = 0
                print(
                    f"  [batch {batch_num}/{total_batches}] +{len(parsed)} texts "
                    f"({len(texts)}/{request_count} total)"
                )
            else:
                consecutive_failures += 1
                print(f"  [batch {batch_num}/{total_batches}] empty response")
        except (RuntimeError, json.JSONDecodeError) as exc:
            print(f"  [batch {batch_num}/{total_batches}] failed: {exc}")
            consecutive_failures += 1

    if not texts:
        raise RuntimeError(
            f"Ollama failed to generate hard negatives after "
            f"{_MAX_RETRIES} consecutive failed attempts"
        )

    # Build hard negative Examples per split
    result = list(examples)
    text_idx = 0

    for split in ("train", "validation", "test"):
        needed = hn_counts.get(split, 0)
        if needed == 0:
            continue

        next_num = split_max_num.get(split, 0) + 1
        domains = ["medical", "scheduling", "workplace", "personal"]

        for i in range(needed):
            if text_idx >= len(texts):
                # Reuse texts if we ran out
                text_idx = 0

            result.append(
                Example(
                    id=f"{split}-{next_num:05d}",
                    text=texts[text_idx],
                    labels=frozenset(),
                    risk=RiskLevel.LOW,
                    rationale="",
                    is_hard_negative=True,
                    split=split,
                    domain=domains[i % len(domains)],
                    source="hard-negative",
                )
            )
            next_num += 1
            text_idx += 1

    return result

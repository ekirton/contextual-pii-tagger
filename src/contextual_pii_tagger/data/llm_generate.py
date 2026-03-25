"""LLM-augmented generation (Stage 2) via Ollama.

Spec: specifications/data-generation.md §3
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from contextual_pii_tagger.entities import RiskLevel, SpanLabel
from contextual_pii_tagger.example import VALID_DOMAINS
from contextual_pii_tagger.data.raw_example import RawExample

logger = logging.getLogger(__name__)

from contextual_pii_tagger.data.batch_limits import max_batch_structured
from contextual_pii_tagger.data.cli_utils import call_ollama

_MAX_RETRIES = 5
_BATCH_MULTIPLIER = 1.3  # request more than needed to account for parse failures

_TAXONOMY_DESCRIPTION = "\n".join(
    f"- {label.value}: Tier 2 quasi-identifier category" for label in SpanLabel
)


def build_generation_prompt(count: int, domain: str) -> str:
    """Build the prompt for generating synthetic PII examples."""
    return f"""\
Generate exactly {count} synthetic examples of text that contains contextual \
personally identifiable information (PII) for the "{domain}" domain.

Each example must be a realistic sentence or short paragraph that a person might \
say or write. The text must be entirely fictional and synthetic — do not use real \
people, real locations, or real organizations.

## Taxonomy of PII categories (SpanLabel values)

{_TAXONOMY_DESCRIPTION}

## Output format

Return a JSON object with a single key "examples" containing an array of objects. \
Each object must have these fields:
- "text": string — the synthetic text (non-empty, 1-3 sentences)
- "labels": array of strings — which SpanLabel categories are present in the text
- "risk": string — one of "LOW", "MEDIUM", "HIGH"
- "rationale": string — why this combination of labels creates re-identification risk
- "domain": string — must be "{domain}"

Example output for 2 examples:
{{"examples": [{{"text": "I see Dr. Patel at Riverside Clinic every Tuesday.", "labels": ["WORKPLACE", "ROUTINE"], "risk": "MEDIUM", "rationale": "Specific clinic and schedule narrow identity.", "domain": "{domain}"}}, {{"text": "The new hire in accounting lives near the waterfront.", "labels": ["WORKPLACE", "LOCATION"], "risk": "MEDIUM", "rationale": "Department and neighborhood narrow identity.", "domain": "{domain}"}}]}}

## Risk level rules

- If no labels apply: risk must be "LOW", rationale must be empty string ""
- If risk is "LOW": rationale must be empty string ""
- If risk is "MEDIUM" or "HIGH" and there are 2+ labels: rationale must be non-empty

## Requirements

- All labels must be valid SpanLabel values from the taxonomy above
- Cover diverse label combinations, not just one pattern
- Text must sound natural, not templated
- All content must be synthetic and fictional
- Each element in the "examples" array MUST be a JSON object, never a plain string"""


def parse_llm_response(raw_examples: list[dict[str, Any]]) -> list[RawExample]:
    """Parse LLM output dicts into RawExample records, skipping malformed entries."""
    results: list[RawExample] = []
    for entry in raw_examples:
        if not isinstance(entry, dict):
            logger.warning("Skipping non-dict entry: %s", type(entry).__name__)
            continue
        try:
            text = entry.get("text", "")
            if not text:
                logger.warning("Skipping entry with empty text")
                continue

            raw_labels = entry.get("labels")
            if raw_labels is None:
                logger.warning("Skipping entry missing labels: %s", text[:50])
                continue

            labels = frozenset(SpanLabel(lbl) for lbl in raw_labels)
            risk = RiskLevel(entry.get("risk", "LOW"))
            rationale = entry.get("rationale", "")
            domain = entry.get("domain", "medical")

            if domain not in VALID_DOMAINS:
                logger.warning("Skipping entry with invalid domain: %s", domain)
                continue

            results.append(
                RawExample(
                    text=text,
                    labels=labels,
                    risk=risk,
                    rationale=rationale,
                    is_hard_negative=False,
                    domain=domain,
                    source="llm-augmented",
                )
            )
        except (ValueError, KeyError) as exc:
            logger.warning("Skipping malformed entry: %s", exc)
            continue

    return results


def generate_from_llm(
    count: int,
    model: str = "sonnet",
    seed: int | None = None,
) -> list[RawExample]:
    """Generate synthetic examples using Ollama.

    Distributes work across domains in round-robin fashion, capping each
    LLM call at ``_MAX_BATCH_SIZE`` examples for reliable output.

    Spec: specifications/data-generation.md §3.1
    """
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")

    domains = list(VALID_DOMAINS)
    results: list[RawExample] = []
    consecutive_failures = 0
    batch_cap = max_batch_structured(model)
    print(f"[Stage 2] Generating {count} examples (batch cap: {batch_cap}/call, model: {model})")

    while len(results) < count and consecutive_failures < _MAX_RETRIES:
        remaining = count - len(results)
        per_domain = max(1, remaining // len(domains))
        made_progress = False

        for domain in domains:
            if len(results) >= count:
                break

            # How many we still want from this domain pass
            domain_target = min(per_domain, count - len(results))
            domain_request = int(math.ceil(domain_target * _BATCH_MULTIPLIER))

            # Break into capped batches
            while domain_request > 0 and len(results) < count:
                batch_size = min(domain_request, batch_cap)
                prompt = build_generation_prompt(count=batch_size, domain=domain)

                try:
                    raw = call_ollama(prompt, model)
                    parsed = parse_llm_response(raw)
                    if parsed:
                        results.extend(parsed)
                        made_progress = True
                        print(
                            f"  [{domain}] +{len(parsed)} examples "
                            f"({len(results)}/{count} total)"
                        )
                except (RuntimeError, json.JSONDecodeError) as exc:
                    print(f"  [{domain}] batch failed (requested {batch_size}): {exc}")
                    break  # move to next domain on failure

                domain_request -= batch_size

        if not made_progress:
            consecutive_failures += 1
        else:
            consecutive_failures = 0

    if not results:
        raise RuntimeError(
            f"Ollama failed to generate any examples after {_MAX_RETRIES} "
            "consecutive failed rounds"
        )

    return results[:count]

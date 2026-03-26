"""Auto-labeling validation (Stage 3) via Ollama.

Spec: specifications/data-generation.md §4
"""

from __future__ import annotations

import json
import logging
from typing import Any

from contextual_pii_tagger.entities import RiskLevel, SpanLabel
from contextual_pii_tagger.example import Example

from contextual_pii_tagger.data.batch_limits import max_batch_validation
from contextual_pii_tagger.data.cli_utils import call_ollama

logger = logging.getLogger(__name__)

_TAXONOMY_DESCRIPTION = "\n".join(
    f"- {label.value}" for label in SpanLabel
)


def build_validation_prompt(examples: list[Example]) -> str:
    """Build the prompt for validating labels on a batch of examples."""
    examples_json = json.dumps(
        [{"id": ex.id, "text": ex.text,
          "labels": sorted(l.value for l in ex.labels),
          "risk": ex.risk.value, "rationale": ex.rationale}
         for ex in examples],
        indent=2,
    )

    return f"""\
You are a PII labeling validator. Review each example below and verify that the \
labels, risk level, and rationale are correct.

## Valid SpanLabel values (Tier 2 quasi-identifiers)

{_TAXONOMY_DESCRIPTION}

## Risk level rules

- Empty labels → risk must be "LOW", rationale must be ""
- Risk "LOW" → rationale must be ""
- Risk "MEDIUM" or "HIGH" → rationale must be non-empty

## Examples to validate

{examples_json}

## Instructions

Return a JSON object with a single key "results" containing an array of objects. \
Each object must have:
- "id": the example's id (unchanged)
- "labels": corrected array of SpanLabel strings (or same if correct)
- "risk": corrected risk level string (or same if correct)
- "rationale": corrected or generated rationale (must satisfy risk rules above)
- "valid": true if the example is usable (even if corrected), false if it should be removed

Example output for 1 example:
{{"results": [{{"id": "train-00001", "labels": ["LOCATION", "ROUTINE"], "risk": "MEDIUM", "rationale": "Location and schedule narrow identity.", "valid": true}}]}}

Return one entry per input example. Each element MUST be a JSON object, never a plain string."""


def _apply_entry(original: Example, entry: dict[str, Any]) -> Example | None:
    """Apply a single validation entry to an example. Returns None if invalid."""
    if not entry.get("valid", True):
        logger.info("Removing invalid example: %s", original.id)
        return None

    try:
        labels = frozenset(SpanLabel(lbl) for lbl in entry.get("labels", []))
        risk = RiskLevel(entry.get("risk", original.risk.value))
        rationale = entry.get("rationale", original.rationale)

        return Example(
            id=original.id,
            text=original.text,
            labels=labels,
            risk=risk,
            rationale=rationale,
            is_hard_negative=original.is_hard_negative,
            split=original.split,
            domain=original.domain,
            source=original.source,
        )
    except (ValueError, KeyError) as exc:
        logger.warning("Skipping malformed validation entry: %s", exc)
        return None


def parse_validation_response(
    examples: list[Example],
    response: list[dict[str, Any]],
) -> list[Example]:
    """Apply validation corrections, removing invalid examples.

    Matching strategy (spec §4.1 Response matching):
    1. ID match (primary): look up entry's id in the input batch.
    2. Positional fallback: if response length == input length and ALL ids
       fail to match, map entries by position.
    3. Unmatched entries are discarded; unmatched input examples are removed.
    """
    # Filter to dict entries only
    dict_entries: list[dict[str, Any]] = []
    for entry in response:
        if isinstance(entry, dict):
            dict_entries.append(entry)
        else:
            logger.warning("Skipping non-dict validation entry: %s", type(entry).__name__)

    examples_by_id = {ex.id: ex for ex in examples}

    # Check if any IDs match
    id_match_count = sum(
        1 for entry in dict_entries if entry.get("id") in examples_by_id
    )

    # Positional fallback: all IDs missed and response length matches input
    if id_match_count == 0 and len(dict_entries) == len(examples):
        logger.info(
            "All %d response IDs unrecognized; using positional fallback",
            len(dict_entries),
        )
        results: list[Example] = []
        for original, entry in zip(examples, dict_entries):
            corrected = _apply_entry(original, entry)
            if corrected is not None:
                results.append(corrected)
        return results

    # Primary path: match by ID
    results = []
    for entry in dict_entries:
        ex_id = entry.get("id")
        if ex_id not in examples_by_id:
            logger.warning("Validation response references unknown id: %s", ex_id)
            continue

        corrected = _apply_entry(examples_by_id[ex_id], entry)
        if corrected is not None:
            results.append(corrected)

    return results


def validate_labels(
    examples: list[Example],
    model: str = "qwen2.5:3b",
) -> list[Example]:
    """Validate and correct labels using a second LLM pass.

    Template-sourced examples pass through unchanged. Only LLM-sourced
    examples are sent to the model for validation.

    Spec: specifications/data-generation.md §4.1
    """
    if not examples:
        return []

    # Separate template and LLM-sourced examples
    template_examples = [ex for ex in examples if ex.source == "template"]
    llm_examples = [ex for ex in examples if ex.source != "template"]

    if not llm_examples:
        return list(examples)

    max_consecutive_failures = 5
    batch_size = max_batch_validation(model)
    total_batches = (len(llm_examples) + batch_size - 1) // batch_size
    print(
        f"[Stage 3] Validating {len(llm_examples)} LLM-sourced examples "
        f"({total_batches} batches of {batch_size}, model: {model}), "
        f"passing through {len(template_examples)} template examples"
    )
    validated_llm: list[Example] = []
    consecutive_failures = 0
    i = 0

    while i < len(llm_examples):
        batch = llm_examples[i : i + batch_size]
        batch_num = i // batch_size + 1
        prompt = build_validation_prompt(batch)

        try:
            response = call_ollama(prompt, model)
        except (RuntimeError, json.JSONDecodeError) as exc:
            consecutive_failures += 1
            print(f"  [batch {batch_num}/{total_batches}] failed: {exc}")
            if consecutive_failures >= max_consecutive_failures:
                raise RuntimeError(
                    f"Stage 3 aborted: {consecutive_failures} consecutive "
                    f"batch failures with no progress"
                ) from exc
            continue

        consecutive_failures = 0
        validated = parse_validation_response(batch, response)
        validated_llm.extend(validated)
        removed = len(batch) - len(validated)
        msg = f"  [batch {batch_num}/{total_batches}] validated {len(validated)}/{len(batch)}"
        if removed:
            msg += f" ({removed} removed)"
        print(f"{msg} ({len(validated_llm)}/{len(llm_examples)} total)")
        i += batch_size

    return template_examples + validated_llm

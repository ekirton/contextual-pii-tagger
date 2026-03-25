"""Generation pipeline orchestrator (T-14).

Spec: specifications/data-generation.md §1
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from contextual_pii_tagger.example import Example
from contextual_pii_tagger.data.raw_example import RawExample
from contextual_pii_tagger.data.templates import generate_from_templates
from contextual_pii_tagger.data.llm_generate import generate_from_llm
from contextual_pii_tagger.data.validate_labels import validate_labels
from contextual_pii_tagger.data.hard_negatives import inject_hard_negatives
from contextual_pii_tagger.data.dataset_io import append_dataset, dataset_stats, write_dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for the generation pipeline."""

    templates_dir: str
    total_count: int = 20000
    template_fraction: float = 0.5
    hard_negative_ratio: float = 0.10
    seed: int | None = None
    model: str = "qwen2.5:3b"
    output_dir: str = "data/"

    def __post_init__(self) -> None:
        if self.total_count <= 0:
            raise ValueError(
                f"total_count must be > 0, got {self.total_count}"
            )
        if not (0.0 < self.template_fraction < 1.0):
            raise ValueError(
                f"template_fraction must be in (0.0, 1.0), got {self.template_fraction}"
            )
        if not (0.0 <= self.hard_negative_ratio < 1.0):
            raise ValueError(
                f"hard_negative_ratio must be in [0.0, 1.0), got {self.hard_negative_ratio}"
            )


def _stratum_key(raw: RawExample) -> str:
    """Return a stratification key combining domain and risk level."""
    return f"{raw.domain}:{raw.risk.value}"


def assign_splits_and_ids(
    raw_examples: list[RawExample],
    seed: int | None = None,
    id_offset_by_split: dict[str, int] | None = None,
) -> list[Example]:
    """Stratified shuffle: assign splits (80/10/10) preserving domain × risk distribution.

    Groups examples into strata by (domain, risk), then independently shuffles
    and splits each stratum so that every split receives a proportional share of
    each combination.  Within each split the examples are shuffled again to
    remove any residual ordering.

    If *id_offset_by_split* is provided, IDs start after the given max ID per
    split (for append/resume workflows).
    """
    rng = random.Random(seed)
    offsets = id_offset_by_split or {}

    # Group indices by stratum
    strata: dict[str, list[int]] = {}
    for idx, raw in enumerate(raw_examples):
        key = _stratum_key(raw)
        strata.setdefault(key, []).append(idx)

    # Assign splits per stratum
    split_map: dict[int, str] = {}
    for key in sorted(strata):
        indices = strata[key]
        rng.shuffle(indices)
        n = len(indices)
        train_end = round(n * 0.8)
        val_end = train_end + round(n * 0.1)

        for i, idx in enumerate(indices):
            if i < train_end:
                split_map[idx] = "train"
            elif i < val_end:
                split_map[idx] = "validation"
            else:
                split_map[idx] = "test"

    # Build Examples grouped by split, then shuffle within each split
    by_split: dict[str, list[tuple[int, RawExample]]] = {
        "train": [], "validation": [], "test": [],
    }
    for idx, raw in enumerate(raw_examples):
        by_split[split_map[idx]].append((idx, raw))

    results: list[Example] = []
    for split in ("train", "validation", "test"):
        items = by_split[split]
        rng.shuffle(items)
        start = offsets.get(split, 0) + 1
        for counter, (_idx, raw) in enumerate(items, start=start):
            results.append(
                Example(
                    id=f"{split}-{counter:05d}",
                    text=raw.text,
                    labels=raw.labels,
                    risk=raw.risk,
                    rationale=raw.rationale,
                    is_hard_negative=raw.is_hard_negative,
                    split=split,
                    domain=raw.domain,
                    source=raw.source,
                )
            )

    return results


def generate_dataset(config: GenerationConfig) -> list[Example]:
    """Orchestrate all five stages of the data generation pipeline.

    If existing JSONL files are found in ``config.output_dir``, the pipeline
    resumes by generating only the remaining examples and appending them.

    Spec: specifications/data-generation.md §1.1
    """
    # Check for existing dataset to resume from
    existing = dataset_stats(config.output_dir)
    existing_non_hn = existing.non_hard_negatives if existing else 0
    existing_total = existing.total if existing else 0
    id_offsets = existing.max_id_by_split if existing else {}

    if existing:
        print(
            f"Found existing dataset: {existing_total} examples "
            f"({existing_non_hn} non-HN, {existing.hard_negatives} HN)"
        )

    # Compute counts (excluding hard negatives)
    non_hn_target = round(config.total_count * (1 - config.hard_negative_ratio))
    remaining_non_hn = max(0, non_hn_target - existing_non_hn)

    if remaining_non_hn == 0:
        print(f"Target of {non_hn_target} non-HN examples already reached. Nothing to generate.")
        return []

    template_count = round(remaining_non_hn * config.template_fraction)
    llm_count = remaining_non_hn - template_count

    print(
        f"Pipeline: target_total={config.total_count}, "
        f"existing={existing_total}, generating={remaining_non_hn} "
        f"(templates={template_count}, llm={llm_count})"
    )

    max_recovery_attempts = 3
    examples: list[Example] = []
    generation_remaining = remaining_non_hn

    for attempt in range(1, max_recovery_attempts + 1):
        t_count = round(generation_remaining * config.template_fraction)
        l_count = generation_remaining - t_count

        # Ensure at least 1 for each source when both would be needed,
        # but skip a source entirely when rounding yields 0.
        # Stage 1: Template-based generation
        if t_count > 0:
            logger.info("Stage 1: Generating %d examples from templates...", t_count)
            template_examples = generate_from_templates(
                config.templates_dir, t_count, seed=config.seed
            )
        else:
            logger.info("Stage 1: Skipping templates (count rounds to 0)")
            template_examples = []

        # Stage 2: LLM-augmented generation
        if l_count > 0:
            logger.info("Stage 2: Generating %d examples from LLM...", l_count)
            llm_examples = generate_from_llm(
                count=l_count, model=config.model, seed=config.seed
            )
        else:
            logger.info("Stage 2: Skipping LLM generation (count rounds to 0)")
            llm_examples = []

        # Combine and assign splits/IDs (continuing from existing max IDs)
        all_raw = template_examples + llm_examples
        # Compute ID offsets from both existing dataset and already-validated examples
        current_offsets = dict(id_offsets)
        for ex in examples:
            num = int(ex.id.rsplit("-", 1)[1])
            current_offsets[ex.split] = max(current_offsets.get(ex.split, 0), num)

        logger.info("Assigning splits and IDs to %d examples...", len(all_raw))
        new_examples = assign_splits_and_ids(
            all_raw, seed=config.seed, id_offset_by_split=current_offsets,
        )

        # Stage 3: Auto-labeling validation
        logger.info("Stage 3: Validating labels on %d examples...", len(new_examples))
        validated = validate_labels(new_examples, model=config.model)
        examples.extend(validated)

        deficit = remaining_non_hn - len(examples)
        if deficit <= 0:
            break

        logger.info(
            "Recovery attempt %d/%d: %d examples still needed",
            attempt, max_recovery_attempts, deficit,
        )
        generation_remaining = deficit

    if len(examples) < remaining_non_hn:
        raise RuntimeError(
            f"Could not reach non-HN target after {max_recovery_attempts} "
            f"recovery attempts: have {len(examples)}, need {remaining_non_hn}"
        )

    # Stage 4: Hard negative injection
    logger.info("Stage 4: Injecting hard negatives (ratio=%.2f)...", config.hard_negative_ratio)
    examples = inject_hard_negatives(
        examples, ratio=config.hard_negative_ratio, model=config.model
    )

    # Write output — append if existing, overwrite if fresh
    if existing:
        logger.info("Appending to existing dataset in %s...", config.output_dir)
        append_dataset(examples, config.output_dir)
    else:
        logger.info("Writing dataset to %s...", config.output_dir)
        write_dataset(examples, config.output_dir)

    final = dataset_stats(config.output_dir)
    print(f"Done. Dataset now has {final.total} examples ({final.hard_negatives} HN).")
    return examples

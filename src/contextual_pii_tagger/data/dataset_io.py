"""Dataset I/O: JSONL read/write.

Spec: specifications/data-generation.md §7
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from contextual_pii_tagger.example import Example

_SPLIT_ORDER = ("train", "validation", "test")


@dataclass(frozen=True)
class DatasetStats:
    """Summary of an existing dataset on disk."""

    total: int
    by_split: dict[str, int]
    max_id_by_split: dict[str, int]
    hard_negatives: int

    @property
    def non_hard_negatives(self) -> int:
        return self.total - self.hard_negatives


def dataset_stats(output_dir: str | Path) -> DatasetStats | None:
    """Read existing JSONL files and return counts, or None if no files exist."""
    output_dir = Path(output_dir)
    by_split: dict[str, int] = {}
    max_id: dict[str, int] = {}
    hard_negatives = 0
    found = False

    for split in _SPLIT_ORDER:
        path = output_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        found = True
        count = 0
        max_num = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                count += 1
                num = int(d["id"].split("-", 1)[1])
                max_num = max(max_num, num)
                if d.get("is_hard_negative", False):
                    hard_negatives += 1
        by_split[split] = count
        max_id[split] = max_num

    if not found:
        return None

    return DatasetStats(
        total=sum(by_split.values()),
        by_split=by_split,
        max_id_by_split=max_id,
        hard_negatives=hard_negatives,
    )


def write_dataset(examples: list[Example], output_dir: str | Path) -> None:
    """Write examples to JSONL files grouped by split (overwrites)."""
    if not examples:
        raise ValueError("Cannot write empty example list")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[Example]] = {s: [] for s in _SPLIT_ORDER}
    for ex in examples:
        by_split[ex.split].append(ex)

    for split in _SPLIT_ORDER:
        path = output_dir / f"{split}.jsonl"
        with open(path, "w") as f:
            for ex in by_split[split]:
                f.write(json.dumps(ex.to_dict()) + "\n")


def append_dataset(examples: list[Example], output_dir: str | Path) -> None:
    """Append examples to existing JSONL files grouped by split."""
    if not examples:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[Example]] = {s: [] for s in _SPLIT_ORDER}
    for ex in examples:
        by_split[ex.split].append(ex)

    for split in _SPLIT_ORDER:
        if not by_split[split]:
            continue
        path = output_dir / f"{split}.jsonl"
        with open(path, "a") as f:
            for ex in by_split[split]:
                f.write(json.dumps(ex.to_dict()) + "\n")


def read_dataset(input_dir: str | Path) -> list[Example]:
    """Read examples from JSONL files in split order."""
    input_dir = Path(input_dir)
    results: list[Example] = []
    found = False

    for split in _SPLIT_ORDER:
        path = input_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        found = True
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(Example.from_dict(json.loads(line)))

    if not found:
        raise ValueError(f"No JSONL files found in {input_dir}")

    return results

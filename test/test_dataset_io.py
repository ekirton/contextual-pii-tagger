"""Tests for dataset I/O (write_dataset, read_dataset).

All expectations derived from specifications/data-generation.md §7.
"""

import json
from pathlib import Path

import pytest

from contextual_pii_tagger.entities import RiskLevel, SpanLabel
from contextual_pii_tagger.example import Example
from contextual_pii_tagger.data.dataset_io import read_dataset, write_dataset


def _make_example(split: str, num: int, domain: str = "medical") -> Example:
    return Example(
        id=f"{split}-{num:05d}",
        text=f"Example text for {split} {num}",
        labels=frozenset({SpanLabel.LOCATION}),
        risk=RiskLevel.MEDIUM,
        rationale="",
        is_hard_negative=False,
        split=split,
        domain=domain,
        source="template",
    )


def _make_dataset(train: int = 4, val: int = 1, test: int = 1) -> list[Example]:
    examples = []
    for i in range(1, train + 1):
        examples.append(_make_example("train", i))
    for i in range(1, val + 1):
        examples.append(_make_example("validation", i))
    for i in range(1, test + 1):
        examples.append(_make_example("test", i))
    return examples


class TestWriteDataset:
    """Tests for write_dataset."""

    def test_writes_three_files(self, tmp_path: Path):
        write_dataset(_make_dataset(), tmp_path)
        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "validation.jsonl").exists()
        assert (tmp_path / "test.jsonl").exists()

    def test_correct_line_counts(self, tmp_path: Path):
        write_dataset(_make_dataset(train=8, val=2, test=2), tmp_path)
        assert len((tmp_path / "train.jsonl").read_text().strip().split("\n")) == 8
        assert len((tmp_path / "validation.jsonl").read_text().strip().split("\n")) == 2
        assert len((tmp_path / "test.jsonl").read_text().strip().split("\n")) == 2

    def test_each_line_is_valid_json(self, tmp_path: Path):
        write_dataset(_make_dataset(), tmp_path)
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
            for line in (tmp_path / name).read_text().strip().split("\n"):
                record = json.loads(line)
                assert isinstance(record, dict)
                assert "id" in record

    def test_each_line_deserializes_to_example(self, tmp_path: Path):
        write_dataset(_make_dataset(), tmp_path)
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
            for line in (tmp_path / name).read_text().strip().split("\n"):
                ex = Example.from_dict(json.loads(line))
                assert isinstance(ex, Example)

    def test_empty_list_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="empty"):
            write_dataset([], tmp_path)

    def test_creates_output_dir(self, tmp_path: Path):
        out = tmp_path / "subdir" / "output"
        write_dataset(_make_dataset(), out)
        assert (out / "train.jsonl").exists()


class TestReadDataset:
    """Tests for read_dataset."""

    def test_round_trip(self, tmp_path: Path):
        original = _make_dataset()
        write_dataset(original, tmp_path)
        loaded = read_dataset(tmp_path)
        assert len(loaded) == len(original)
        original_by_id = {ex.id: ex for ex in original}
        for ex in loaded:
            assert ex == original_by_id[ex.id]

    def test_order_train_val_test(self, tmp_path: Path):
        write_dataset(_make_dataset(train=2, val=2, test=2), tmp_path)
        loaded = read_dataset(tmp_path)
        splits = [ex.split for ex in loaded]
        assert splits == ["train", "train", "validation", "validation", "test", "test"]

    def test_partial_files(self, tmp_path: Path):
        """Works when only some split files exist."""
        examples = [_make_example("train", 1)]
        write_dataset(examples, tmp_path)
        # Remove validation and test files (they'll be empty)
        loaded = read_dataset(tmp_path)
        train_examples = [ex for ex in loaded if ex.split == "train"]
        assert len(train_examples) == 1

    def test_empty_dir_raises(self, tmp_path: Path):
        with pytest.raises(ValueError):
            read_dataset(tmp_path)

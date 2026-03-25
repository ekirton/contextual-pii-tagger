"""Tests for the generation pipeline orchestrator (T-14).

All expectations derived from specifications/data-generation.md §1.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from contextual_pii_tagger.entities import RiskLevel, SpanLabel
from contextual_pii_tagger.example import Example
from contextual_pii_tagger.data.raw_example import RawExample
from contextual_pii_tagger.data.generate import (
    GenerationConfig,
    generate_dataset,
    assign_splits_and_ids,
)


def _make_raw_example(domain: str = "medical", label: SpanLabel = SpanLabel.LOCATION) -> RawExample:
    return RawExample(
        text=f"Example text for {domain}",
        labels=frozenset({label}),
        risk=RiskLevel.MEDIUM,
        rationale="",
        is_hard_negative=False,
        domain=domain,
        source="template",
    )


def _make_raw_examples(count: int) -> list[RawExample]:
    domains = ["medical", "scheduling", "workplace", "personal"]
    labels = [SpanLabel.LOCATION, SpanLabel.ROUTINE, SpanLabel.WORKPLACE, SpanLabel.DEMOGRAPHIC]
    return [
        _make_raw_example(domains[i % len(domains)], labels[i % len(labels)])
        for i in range(count)
    ]


# ── GenerationConfig ─────────────────────────────────────────────────────


class TestGenerationConfig:
    """Tests for config validation."""

    def test_default_config(self, tmp_path: Path):
        config = GenerationConfig(templates_dir=str(tmp_path))
        assert config.total_count == 20000
        assert config.template_fraction == 0.5
        assert config.hard_negative_ratio == 0.10
        assert config.model == "qwen2.5:3b"

    def test_custom_count(self, tmp_path: Path):
        config = GenerationConfig(templates_dir=str(tmp_path), total_count=100)
        assert config.total_count == 100

    def test_zero_count_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="total_count"):
            GenerationConfig(templates_dir=str(tmp_path), total_count=0)

    def test_invalid_template_fraction_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="template_fraction"):
            GenerationConfig(templates_dir=str(tmp_path), template_fraction=1.5)

    def test_invalid_hard_negative_ratio_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="hard_negative_ratio"):
            GenerationConfig(templates_dir=str(tmp_path), hard_negative_ratio=1.0)


# ── assign_splits_and_ids ────────────────────────────────────────────────


class TestAssignSplitsAndIds:
    """Tests for split assignment and ID generation."""

    def test_correct_split_ratios(self):
        raw = _make_raw_examples(100)
        examples = assign_splits_and_ids(raw, seed=42)
        splits = {"train": 0, "validation": 0, "test": 0}
        for ex in examples:
            splits[ex.split] += 1
        assert len(examples) == 100
        # Stratified splitting applies 80/10/10 per stratum; per-stratum
        # rounding may shift counts by a small amount.
        assert abs(splits["train"] - 80) <= 4
        assert abs(splits["validation"] - 10) <= 4
        assert abs(splits["test"] - 10) <= 4

    def test_ids_follow_pattern(self):
        raw = _make_raw_examples(20)
        examples = assign_splits_and_ids(raw, seed=42)
        import re
        pattern = re.compile(r"^(train|validation|test)-\d{5}$")
        for ex in examples:
            assert pattern.match(ex.id), f"Invalid id: {ex.id}"

    def test_ids_are_unique(self):
        raw = _make_raw_examples(100)
        examples = assign_splits_and_ids(raw, seed=42)
        ids = [ex.id for ex in examples]
        assert len(ids) == len(set(ids))

    def test_deterministic_with_seed(self):
        raw = _make_raw_examples(50)
        a = assign_splits_and_ids(raw, seed=99)
        b = assign_splits_and_ids(raw, seed=99)
        for x, y in zip(a, b):
            assert x.id == y.id
            assert x.split == y.split

    def test_preserves_fields(self):
        raw = _make_raw_examples(10)
        examples = assign_splits_and_ids(raw, seed=42)
        for ex, raw_ex in zip(
            sorted(examples, key=lambda e: e.text),
            sorted(raw, key=lambda e: e.text),
        ):
            assert ex.text == raw_ex.text
            assert ex.labels == raw_ex.labels
            assert ex.risk == raw_ex.risk
            assert ex.domain == raw_ex.domain
            assert ex.source == raw_ex.source

    def test_all_results_are_examples(self):
        raw = _make_raw_examples(10)
        examples = assign_splits_and_ids(raw, seed=42)
        assert all(isinstance(ex, Example) for ex in examples)


# ── generate_dataset (integration with mocks) ───────────────────────────


def _mock_llm_response(count: int) -> list[dict]:
    domains = ["medical", "scheduling", "workplace", "personal"]
    all_labels = list(SpanLabel)
    examples = []
    for i in range(count):
        label = all_labels[i % len(all_labels)]
        examples.append({
            "text": f"LLM generated text {i}",
            "labels": [label.value],
            "risk": "MEDIUM",
            "rationale": "",
            "domain": domains[i % len(domains)],
        })
    return examples


def _mock_validation_response(examples: list[Example]) -> list[dict]:
    return [
        {"id": ex.id, "labels": sorted(l.value for l in ex.labels),
         "risk": ex.risk.value, "rationale": ex.rationale, "valid": True}
        for ex in examples
    ]


def _mock_hard_negative_texts(count: int) -> list[str]:
    return [f"George Washington crossed the Delaware (example {i})." for i in range(count)]


def _setup_template_dir(tmp_path: Path) -> Path:
    """Create minimal template files for testing."""
    import yaml
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    for domain in ("medical", "scheduling", "workplace", "personal"):
        data = {
            "domain": domain,
            "patterns": [
                {
                    "text": "I work at {COMPANY} in {CITY}",
                    "labels": ["WORKPLACE", "LOCATION"],
                    "risk": "MEDIUM",
                    "rationale_template": f"Workplace and location in {domain}.",
                },
            ],
        }
        (templates_dir / f"{domain}.yaml").write_text(yaml.dump(data))
    return templates_dir


def _make_ollama_response(content: list | dict) -> MagicMock:
    """Build a mock urllib response for Ollama /api/chat."""
    body = json.dumps({
        "message": {"role": "assistant", "content": json.dumps(content)},
    }).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _extract_examples_from_validation_prompt(prompt: str) -> list[dict]:
    """Extract the examples JSON array from a validation prompt."""
    marker = "## Examples to validate"
    section_start = prompt.index(marker) + len(marker)
    # Find the next section header to bound the search
    next_section = prompt.index("##", section_start)
    section = prompt[section_start:next_section]
    return json.loads(section.strip())


def _ollama_dispatcher(req, **kwargs):
    """Dispatch urlopen calls based on prompt content in the request body."""
    payload = json.loads(req.data)
    prompt = payload["messages"][0]["content"]

    if "hard negative" in prompt.lower() or "not personally" in prompt.lower():
        return _make_ollama_response(_mock_hard_negative_texts(30))
    elif "labeling validator" in prompt.lower() or "verify" in prompt.lower():
        try:
            examples_data = _extract_examples_from_validation_prompt(prompt)
            response = [
                {"id": ex["id"], "labels": ex["labels"],
                 "risk": ex["risk"], "rationale": ex["rationale"], "valid": True}
                for ex in examples_data
            ]
        except (ValueError, json.JSONDecodeError):
            response = []
        return _make_ollama_response(response)
    else:
        return _make_ollama_response(_mock_llm_response(50))


class TestGenerateDataset:
    """Integration tests for the full pipeline with mocked LLM calls."""

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen",
           side_effect=_ollama_dispatcher)
    def test_produces_correct_total_count(self, mock_urlopen, tmp_path):
        templates_dir = _setup_template_dir(tmp_path)
        config = GenerationConfig(
            templates_dir=str(templates_dir),
            total_count=100,
            seed=42,
            output_dir=str(tmp_path / "output"),
        )
        dataset = generate_dataset(config)
        assert len(dataset) == 100

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen",
           side_effect=_ollama_dispatcher)
    def test_split_ratios(self, mock_urlopen, tmp_path):
        templates_dir = _setup_template_dir(tmp_path)
        config = GenerationConfig(
            templates_dir=str(templates_dir),
            total_count=100,
            seed=42,
            output_dir=str(tmp_path / "output"),
        )
        dataset = generate_dataset(config)
        non_hn = [ex for ex in dataset if not ex.is_hard_negative]
        splits = {"train": 0, "validation": 0, "test": 0}
        for ex in non_hn:
            splits[ex.split] += 1
        total = len(non_hn)
        # Stratified splitting: approximate 80/10/10 with per-stratum rounding
        assert abs(splits["train"] / total - 0.8) < 0.05
        assert abs(splits["validation"] / total - 0.1) < 0.05
        assert abs(splits["test"] / total - 0.1) < 0.05

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen",
           side_effect=_ollama_dispatcher)
    def test_hard_negative_ratio(self, mock_urlopen, tmp_path):
        templates_dir = _setup_template_dir(tmp_path)
        config = GenerationConfig(
            templates_dir=str(templates_dir),
            total_count=100,
            hard_negative_ratio=0.10,
            seed=42,
            output_dir=str(tmp_path / "output"),
        )
        dataset = generate_dataset(config)
        hn_count = sum(1 for ex in dataset if ex.is_hard_negative)
        assert hn_count == 10

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen",
           side_effect=_ollama_dispatcher)
    def test_writes_output_files(self, mock_urlopen, tmp_path):
        templates_dir = _setup_template_dir(tmp_path)
        output_dir = tmp_path / "output"
        config = GenerationConfig(
            templates_dir=str(templates_dir),
            total_count=100,
            seed=42,
            output_dir=str(output_dir),
        )
        generate_dataset(config)
        assert (output_dir / "train.jsonl").exists()
        assert (output_dir / "validation.jsonl").exists()
        assert (output_dir / "test.jsonl").exists()

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_recovers_from_validation_loss(self, mock_urlopen, tmp_path):
        """Pipeline must regenerate examples lost during validation (§1.1, §4.1)."""
        templates_dir = _setup_template_dir(tmp_path)
        output_dir = tmp_path / "output"
        config = GenerationConfig(
            templates_dir=str(templates_dir),
            total_count=100,
            seed=42,
            output_dir=str(output_dir),
        )

        call_count = {"validation": 0}

        def _dispatcher_with_validation_loss(req, **kwargs):
            payload = json.loads(req.data)
            prompt = payload["messages"][0]["content"]

            if "hard negative" in prompt.lower() or "not personally" in prompt.lower():
                return _make_ollama_response(_mock_hard_negative_texts(30))
            elif "labeling validator" in prompt.lower() or "verify" in prompt.lower():
                call_count["validation"] += 1
                try:
                    examples_data = _extract_examples_from_validation_prompt(prompt)
                    if call_count["validation"] == 1:
                        # First validation pass: drop ~20% of examples
                        keep = examples_data[: int(len(examples_data) * 0.8)]
                    else:
                        keep = examples_data
                    response = [
                        {"id": ex["id"], "labels": ex["labels"],
                         "risk": ex["risk"], "rationale": ex["rationale"], "valid": True}
                        for ex in keep
                    ]
                except (ValueError, json.JSONDecodeError):
                    response = []
                return _make_ollama_response(response)
            else:
                return _make_ollama_response(_mock_llm_response(50))

        mock_urlopen.side_effect = _dispatcher_with_validation_loss
        dataset = generate_dataset(config)
        assert len(dataset) == 100

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_recovery_with_single_example_remaining(self, mock_urlopen, tmp_path):
        """Recovery must not crash when only 1 example remains (t_count rounds to 0)."""
        templates_dir = _setup_template_dir(tmp_path)
        output_dir = tmp_path / "output"
        config = GenerationConfig(
            templates_dir=str(templates_dir),
            total_count=20,
            seed=42,
            output_dir=str(output_dir),
        )

        call_count = {"validation": 0}

        def _dispatcher_leaving_one_short(req, **kwargs):
            payload = json.loads(req.data)
            prompt = payload["messages"][0]["content"]

            if "hard negative" in prompt.lower() or "not personally" in prompt.lower():
                return _make_ollama_response(_mock_hard_negative_texts(30))
            elif "labeling validator" in prompt.lower() or "verify" in prompt.lower():
                call_count["validation"] += 1
                try:
                    examples_data = _extract_examples_from_validation_prompt(prompt)
                    if call_count["validation"] == 1:
                        # First validation: drop exactly 1 example to leave deficit=1
                        keep = examples_data[1:]
                    else:
                        keep = examples_data
                    response = [
                        {"id": ex["id"], "labels": ex["labels"],
                         "risk": ex["risk"], "rationale": ex["rationale"], "valid": True}
                        for ex in keep
                    ]
                except (ValueError, json.JSONDecodeError):
                    response = []
                return _make_ollama_response(response)
            else:
                return _make_ollama_response(_mock_llm_response(50))

        mock_urlopen.side_effect = _dispatcher_leaving_one_short
        dataset = generate_dataset(config)
        assert len(dataset) == 20

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen",
           side_effect=_ollama_dispatcher)
    def test_all_examples_valid(self, mock_urlopen, tmp_path):
        templates_dir = _setup_template_dir(tmp_path)
        config = GenerationConfig(
            templates_dir=str(templates_dir),
            total_count=100,
            seed=42,
            output_dir=str(tmp_path / "output"),
        )
        dataset = generate_dataset(config)
        for ex in dataset:
            assert isinstance(ex, Example)

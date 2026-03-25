"""Tests for template-based generation.

All expectations derived from specifications/data-generation.md §2.1-2.2.
"""

import re
from pathlib import Path

import pytest
import yaml

from contextual_pii_tagger.entities import (
    RiskLevel,
    SpanLabel,
)
from contextual_pii_tagger.data.raw_example import RawExample
from contextual_pii_tagger.data.templates import (
    DomainTemplateFile,
    TemplatePattern,
    fill_slots,
    generate_from_templates,
    load_template_file,
    load_templates,
    SLOT_REGISTRY,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.dump(data, default_flow_style=False))
    return path


def _valid_template_data() -> dict:
    return {
        "domain": "medical",
        "patterns": [
            {
                "text": "I saw my {SPECIALTY} at {HOSPITAL} last {DAY_OF_WEEK}",
                "labels": ["MEDICAL-CONTEXT", "WORKPLACE", "ROUTINE"],
                "risk": "MEDIUM",
                "rationale_template": "Medical specialty and hospital narrow identification.",
            },
            {
                "text": "My {CONDITION} checkup at {CLINIC} is every {FREQUENCY}",
                "labels": ["MEDICAL-CONTEXT", "WORKPLACE", "ROUTINE"],
                "risk": "MEDIUM",
                "rationale_template": "Condition and clinic reveal health status.",
            },
        ],
    }


def _multi_domain_dir(tmp_path: Path) -> Path:
    """Create a directory with 4 minimal domain files."""
    domains = {
        "medical": {
            "domain": "medical",
            "patterns": [
                {
                    "text": "I saw my {SPECIALTY} at {HOSPITAL}",
                    "labels": ["MEDICAL-CONTEXT", "WORKPLACE"],
                    "risk": "MEDIUM",
                    "rationale_template": "Specialty and hospital narrow identity.",
                },
            ],
        },
        "scheduling": {
            "domain": "scheduling",
            "patterns": [
                {
                    "text": "I take the {BUS_ROUTE} at {TIME}",
                    "labels": ["ROUTINE", "LOCATION"],
                    "risk": "MEDIUM",
                    "rationale_template": "Route and time reveal commute pattern.",
                },
            ],
        },
        "workplace": {
            "domain": "workplace",
            "patterns": [
                {
                    "text": "I work at {COMPANY} in {CITY}",
                    "labels": ["WORKPLACE", "LOCATION"],
                    "risk": "MEDIUM",
                    "rationale_template": "Company and city narrow location.",
                },
            ],
        },
        "personal": {
            "domain": "personal",
            "patterns": [
                {
                    "text": "I live in {NEIGHBORHOOD} near {CITY}",
                    "labels": ["LOCATION"],
                    "risk": "LOW",
                    "rationale_template": "",
                },
            ],
        },
    }
    for name, data in domains.items():
        _write_yaml(tmp_path / f"{name}.yaml", data)
    return tmp_path


# ── YAML Loader Tests ────────────────────────────────────────────────────


class TestLoadTemplates:
    """Tests for load_template_file and load_templates."""

    def test_load_single_file(self, tmp_path: Path):
        path = _write_yaml(tmp_path / "medical.yaml", _valid_template_data())
        result = load_template_file(path)
        assert isinstance(result, DomainTemplateFile)
        assert result.domain == "medical"
        assert len(result.patterns) == 2
        assert isinstance(result.patterns[0], TemplatePattern)
        assert SpanLabel.MEDICAL_CONTEXT in result.patterns[0].labels

    def test_invalid_label_raises(self, tmp_path: Path):
        data = _valid_template_data()
        data["patterns"][0]["labels"] = ["NOT-A-LABEL"]
        path = _write_yaml(tmp_path / "bad.yaml", data)
        with pytest.raises(ValueError, match="label"):
            load_template_file(path)

    def test_missing_domain_raises(self, tmp_path: Path):
        data = _valid_template_data()
        del data["domain"]
        path = _write_yaml(tmp_path / "bad.yaml", data)
        with pytest.raises(ValueError, match="domain"):
            load_template_file(path)

    def test_invalid_domain_raises(self, tmp_path: Path):
        data = _valid_template_data()
        data["domain"] = "finance"
        path = _write_yaml(tmp_path / "bad.yaml", data)
        with pytest.raises(ValueError, match="domain"):
            load_template_file(path)

    def test_empty_patterns_raises(self, tmp_path: Path):
        data = _valid_template_data()
        data["patterns"] = []
        path = _write_yaml(tmp_path / "bad.yaml", data)
        with pytest.raises(ValueError, match="pattern"):
            load_template_file(path)

    def test_missing_text_raises(self, tmp_path: Path):
        data = _valid_template_data()
        del data["patterns"][0]["text"]
        path = _write_yaml(tmp_path / "bad.yaml", data)
        with pytest.raises(ValueError, match="text"):
            load_template_file(path)

    def test_missing_labels_raises(self, tmp_path: Path):
        data = _valid_template_data()
        data["patterns"][0]["labels"] = []
        path = _write_yaml(tmp_path / "bad.yaml", data)
        with pytest.raises(ValueError, match="label"):
            load_template_file(path)

    def test_missing_risk_raises(self, tmp_path: Path):
        data = _valid_template_data()
        del data["patterns"][0]["risk"]
        path = _write_yaml(tmp_path / "bad.yaml", data)
        with pytest.raises(ValueError, match="risk"):
            load_template_file(path)

    def test_load_directory(self, tmp_path: Path):
        templates_dir = _multi_domain_dir(tmp_path)
        result = load_templates(templates_dir)
        assert len(result) == 4
        domains = {t.domain for t in result}
        assert domains == {"medical", "scheduling", "workplace", "personal"}


# ── Slot Filling Tests ───────────────────────────────────────────────────


class TestSlotFilling:
    """Tests for fill_slots."""

    def test_fills_all_slots(self):
        from faker import Faker

        fake = Faker()
        fake.seed_instance(42)
        text = "I saw my {SPECIALTY} at {HOSPITAL} last {DAY_OF_WEEK}"
        result = fill_slots(text, fake)
        assert "{" not in result
        assert "}" not in result

    def test_variation(self):
        from faker import Faker

        fake = Faker()
        text = "I work at {COMPANY} in {CITY}"
        results = {fill_slots(text, fake) for _ in range(10)}
        assert len(results) > 1, "Expected variation across fills"

    def test_unknown_slot_raises(self):
        from faker import Faker

        fake = Faker()
        with pytest.raises(KeyError):
            fill_slots("Hello {NONEXISTENT_SLOT}", fake)

    def test_nonempty_after_fill(self):
        from faker import Faker

        fake = Faker()
        fake.seed_instance(42)
        text = "I live in {CITY}"
        result = fill_slots(text, fake)
        assert len(result) > 0


# ── generate_from_templates Integration Tests ────────────────────────────


class TestGenerateFromTemplates:
    """Tests for generate_from_templates."""

    def test_returns_correct_count(self, tmp_path: Path):
        templates_dir = _multi_domain_dir(tmp_path)
        result = generate_from_templates(templates_dir, 20, seed=42)
        assert len(result) == 20

    def test_all_source_template(self, tmp_path: Path):
        templates_dir = _multi_domain_dir(tmp_path)
        result = generate_from_templates(templates_dir, 10, seed=42)
        assert all(ex.source == "template" for ex in result)

    def test_all_not_hard_negative(self, tmp_path: Path):
        templates_dir = _multi_domain_dir(tmp_path)
        result = generate_from_templates(templates_dir, 10, seed=42)
        assert all(not ex.is_hard_negative for ex in result)

    def test_labels_from_template(self, tmp_path: Path):
        """Each result's labels must be a valid frozenset of SpanLabels."""
        templates_dir = _multi_domain_dir(tmp_path)
        result = generate_from_templates(templates_dir, 10, seed=42)
        for ex in result:
            assert isinstance(ex.labels, frozenset)
            for label in ex.labels:
                assert isinstance(label, SpanLabel)

    def test_domains_distributed(self, tmp_path: Path):
        templates_dir = _multi_domain_dir(tmp_path)
        result = generate_from_templates(templates_dir, 20, seed=42)
        domains = {ex.domain for ex in result}
        assert len(domains) >= 2, f"Expected multiple domains, got {domains}"

    def test_count_zero_raises(self, tmp_path: Path):
        templates_dir = _multi_domain_dir(tmp_path)
        with pytest.raises(ValueError, match="count"):
            generate_from_templates(templates_dir, 0, seed=42)

    def test_negative_count_raises(self, tmp_path: Path):
        templates_dir = _multi_domain_dir(tmp_path)
        with pytest.raises(ValueError, match="count"):
            generate_from_templates(templates_dir, -1, seed=42)

    def test_empty_directory_raises(self, tmp_path: Path):
        with pytest.raises(ValueError):
            generate_from_templates(tmp_path, 10, seed=42)

    def test_deterministic_with_seed(self, tmp_path: Path):
        templates_dir = _multi_domain_dir(tmp_path)
        a = generate_from_templates(templates_dir, 10, seed=99)
        b = generate_from_templates(templates_dir, 10, seed=99)
        for x, y in zip(a, b):
            assert x.text == y.text
            assert x.labels == y.labels
            assert x.domain == y.domain

    def test_risk_rationale_invariants(self, tmp_path: Path):
        """Every result satisfies DetectionResult invariants."""
        templates_dir = _multi_domain_dir(tmp_path)
        result = generate_from_templates(templates_dir, 20, seed=42)
        for ex in result:
            if len(ex.labels) == 0:
                assert ex.risk == RiskLevel.LOW
            if ex.risk == RiskLevel.LOW:
                assert ex.rationale == ""
            if ex.risk in (RiskLevel.MEDIUM, RiskLevel.HIGH) and len(ex.labels) >= 2:
                assert ex.rationale != ""

    def test_all_results_are_raw_examples(self, tmp_path: Path):
        templates_dir = _multi_domain_dir(tmp_path)
        result = generate_from_templates(templates_dir, 10, seed=42)
        assert all(isinstance(ex, RawExample) for ex in result)

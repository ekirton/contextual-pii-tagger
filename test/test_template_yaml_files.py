"""Acceptance tests for the four domain YAML template files (T-23).

All expectations derived from specifications/data-generation.md §2.2
and tasks/t23-template-files.md acceptance criteria.
"""

import re
from pathlib import Path

import pytest

from contextual_pii_tagger.entities import RiskLevel, SpanLabel
from contextual_pii_tagger.data.templates import (
    load_template_file,
    load_templates,
    SLOT_REGISTRY,
)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "src" / "contextual_pii_tagger" / "data" / "templates"

EXPECTED_DOMAINS = {"medical", "scheduling", "workplace", "personal"}
ALL_SPAN_LABELS = frozenset(SpanLabel)
SLOT_PATTERN = re.compile(r"\{(\w+)\}")


class TestTemplateYamlFiles:
    """Validate the four domain YAML template artifacts."""

    def test_all_four_files_exist(self):
        for domain in EXPECTED_DOMAINS:
            path = TEMPLATES_DIR / f"{domain}.yaml"
            assert path.exists(), f"Missing template file: {path}"

    def test_each_file_parses(self):
        for domain in EXPECTED_DOMAINS:
            path = TEMPLATES_DIR / f"{domain}.yaml"
            result = load_template_file(path)
            assert result.domain == domain

    def test_min_10_patterns_each(self):
        for domain in EXPECTED_DOMAINS:
            path = TEMPLATES_DIR / f"{domain}.yaml"
            result = load_template_file(path)
            assert len(result.patterns) >= 10, (
                f"{domain}.yaml has {len(result.patterns)} patterns, need >= 10"
            )

    def test_domain_matches_filename(self):
        for domain in EXPECTED_DOMAINS:
            path = TEMPLATES_DIR / f"{domain}.yaml"
            result = load_template_file(path)
            assert result.domain == domain, (
                f"{domain}.yaml has domain={result.domain!r}, expected {domain!r}"
            )

    def test_all_labels_valid(self):
        templates = load_templates(TEMPLATES_DIR)
        for tmpl in templates:
            for pattern in tmpl.patterns:
                for label in pattern.labels:
                    assert isinstance(label, SpanLabel), (
                        f"Invalid label {label!r} in {tmpl.domain}"
                    )

    def test_all_eight_labels_covered(self):
        templates = load_templates(TEMPLATES_DIR)
        all_labels: set[SpanLabel] = set()
        for tmpl in templates:
            for pattern in tmpl.patterns:
                all_labels.update(pattern.labels)
        missing = ALL_SPAN_LABELS - all_labels
        assert not missing, f"Missing SpanLabel coverage: {missing}"

    def test_all_risks_valid(self):
        templates = load_templates(TEMPLATES_DIR)
        for tmpl in templates:
            for pattern in tmpl.patterns:
                assert isinstance(pattern.risk, RiskLevel), (
                    f"Invalid risk {pattern.risk!r} in {tmpl.domain}"
                )

    def test_all_patterns_have_slots(self):
        templates = load_templates(TEMPLATES_DIR)
        for tmpl in templates:
            for pattern in tmpl.patterns:
                slots = SLOT_PATTERN.findall(pattern.text_template)
                assert len(slots) > 0, (
                    f"Pattern in {tmpl.domain} has no slots: {pattern.text_template!r}"
                )

    def test_all_slots_registered(self):
        templates = load_templates(TEMPLATES_DIR)
        for tmpl in templates:
            for pattern in tmpl.patterns:
                slots = SLOT_PATTERN.findall(pattern.text_template)
                for slot in slots:
                    assert slot in SLOT_REGISTRY, (
                        f"Unregistered slot {slot!r} in {tmpl.domain}: "
                        f"{pattern.text_template!r}"
                    )

    def test_rationale_when_multi_label_medium_high(self):
        """Patterns with MEDIUM/HIGH risk and 2+ labels need rationale_template."""
        templates = load_templates(TEMPLATES_DIR)
        for tmpl in templates:
            for pattern in tmpl.patterns:
                if (
                    pattern.risk in (RiskLevel.MEDIUM, RiskLevel.HIGH)
                    and len(pattern.labels) >= 2
                ):
                    assert pattern.rationale_template, (
                        f"Pattern in {tmpl.domain} has {pattern.risk} risk and "
                        f"{len(pattern.labels)} labels but empty rationale_template: "
                        f"{pattern.text_template!r}"
                    )

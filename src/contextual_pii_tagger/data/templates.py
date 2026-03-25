"""Template-based synthetic data generation (Stage 1).

Spec: specifications/data-generation.md §2.1-2.2
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml
from faker import Faker

from contextual_pii_tagger.entities import RiskLevel, SpanLabel
from contextual_pii_tagger.example import VALID_DOMAINS
from contextual_pii_tagger.data.raw_example import RawExample

_SLOT_RE = re.compile(r"\{(\w+)\}")


# ── Slot Registry ────────────────────────────────────────────────────────

_SPECIALTIES = [
    "cardiologist", "dermatologist", "neurologist", "oncologist",
    "orthopedist", "pediatrician", "psychiatrist", "rheumatologist",
    "endocrinologist", "gastroenterologist", "pulmonologist", "urologist",
]

_CONDITIONS = [
    "diabetes", "hypertension", "asthma", "arthritis", "migraine",
    "anxiety", "depression", "thyroid disorder", "anemia", "COPD",
]

_MEDICATIONS = [
    "metformin", "lisinopril", "atorvastatin", "levothyroxine",
    "amlodipine", "omeprazole", "sertraline", "albuterol",
]

_DEPARTMENTS = [
    "Engineering", "Marketing", "Human Resources", "Finance",
    "Sales", "Operations", "Legal", "Product", "Design", "Research",
]

_FREQUENCIES = [
    "every Monday", "every Tuesday", "every Wednesday", "every Thursday",
    "every Friday", "twice a week", "daily", "every other day",
    "three times a week", "weekly",
]

_ETHNICITIES = [
    "Asian", "Black", "Hispanic", "White", "Indigenous",
    "Pacific Islander", "Middle Eastern", "South Asian",
]

_RELIGIONS = [
    "Catholic", "Protestant", "Jewish", "Muslim", "Hindu",
    "Buddhist", "Sikh", "Unitarian",
]

_DEVICE_MODELS = [
    "iPhone 15", "Galaxy S24", "Pixel 8", "iPad Pro",
    "MacBook Air", "Surface Pro", "Fitbit Charge 6", "Apple Watch Ultra",
]

_PASSWORD_HINTS = [
    "my pet's name plus birth year", "favorite color and street number",
    "childhood nickname backwards", "first car model with ZIP code",
]


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]}"


SLOT_REGISTRY: dict[str, Callable[[Faker], str]] = {
    "HOSPITAL": lambda f: f"{f.company()} Hospital",
    "SPECIALTY": lambda f: f.random_element(_SPECIALTIES),
    "DAY_OF_WEEK": lambda f: f.day_of_week(),
    "CLINIC": lambda f: f"{f.company()} Clinic",
    "MEDICATION": lambda f: f.random_element(_MEDICATIONS),
    "CONDITION": lambda f: f.random_element(_CONDITIONS),
    "TIME": lambda f: f.time(pattern="%I:%M %p"),
    "NEIGHBORHOOD": lambda f: f.street_name(),
    "CITY": lambda f: f.city(),
    "LOCATION": lambda f: f.street_address(),
    "SCHOOL": lambda f: f"{f.last_name()} Elementary",
    "BUS_ROUTE": lambda f: f.numerify("Route ##"),
    "COMPANY": lambda f: f.company(),
    "DEPARTMENT": lambda f: f.random_element(_DEPARTMENTS),
    "JOB_TITLE": lambda f: f.job(),
    "BUILDING": lambda f: f"{f.building_number()} {f.street_name()}",
    "FLOOR": lambda f: _ordinal(f.random_int(min=1, max=30)),
    "AGE": lambda f: str(f.random_int(min=18, max=75)),
    "ETHNICITY": lambda f: f.random_element(_ETHNICITIES),
    "RELIGION": lambda f: f.random_element(_RELIGIONS),
    "DEVICE_MODEL": lambda f: f.random_element(_DEVICE_MODELS),
    "DEVICE_ID": lambda f: f.numerify("SN-########"),
    "USERNAME": lambda f: f.user_name(),
    "NAME": lambda f: f.name(),
    "FIRST_NAME": lambda f: f.first_name(),
    "FREQUENCY": lambda f: f.random_element(_FREQUENCIES),
    "STATE": lambda f: f.state(),
    "MONTH": lambda f: f.month_name(),
    "PASSWORD_HINT": lambda f: f.random_element(_PASSWORD_HINTS),
    "YEAR": lambda f: str(f.random_int(min=1, max=30)),
}


# ── Data Classes ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TemplatePattern:
    """One parsed template pattern."""

    text_template: str
    labels: frozenset[SpanLabel]
    risk: RiskLevel
    rationale_template: str


@dataclass(frozen=True)
class DomainTemplateFile:
    """Parsed contents of one domain YAML file."""

    domain: str
    patterns: list[TemplatePattern]


# ── YAML Loading ─────────────────────────────────────────────────────────


def load_template_file(path: Path) -> DomainTemplateFile:
    """Parse and validate a single YAML template file."""
    with open(path) as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict) or "domain" not in data:
        raise ValueError(f"Template file {path} missing 'domain' field")

    domain = data["domain"]
    if domain not in VALID_DOMAINS:
        raise ValueError(
            f"Template file {path} has invalid domain {domain!r}, "
            f"expected one of {VALID_DOMAINS}"
        )

    raw_patterns = data.get("patterns")
    if not raw_patterns:
        raise ValueError(f"Template file {path} has no patterns")

    patterns: list[TemplatePattern] = []
    for i, p in enumerate(raw_patterns):
        if "text" not in p:
            raise ValueError(f"Pattern {i} in {path} missing 'text' field")
        if "risk" not in p:
            raise ValueError(f"Pattern {i} in {path} missing 'risk' field")

        raw_labels = p.get("labels", [])
        if not raw_labels:
            raise ValueError(f"Pattern {i} in {path} has no labels")

        try:
            labels = frozenset(SpanLabel(lbl) for lbl in raw_labels)
        except ValueError as exc:
            raise ValueError(
                f"Pattern {i} in {path} has invalid label: {exc}"
            ) from exc

        try:
            risk = RiskLevel(p["risk"])
        except ValueError as exc:
            raise ValueError(
                f"Pattern {i} in {path} has invalid risk: {exc}"
            ) from exc

        patterns.append(
            TemplatePattern(
                text_template=p["text"],
                labels=labels,
                risk=risk,
                rationale_template=p.get("rationale_template", ""),
            )
        )

    return DomainTemplateFile(domain=domain, patterns=patterns)


def load_templates(templates_dir: Path | str) -> list[DomainTemplateFile]:
    """Load all YAML template files from a directory."""
    templates_dir = Path(templates_dir)
    yaml_files = sorted(templates_dir.glob("*.yaml"))
    if not yaml_files:
        raise ValueError(f"No YAML files found in {templates_dir}")
    return [load_template_file(f) for f in yaml_files]


# ── Slot Filling ─────────────────────────────────────────────────────────


def fill_slots(text_template: str, faker: Faker) -> str:
    """Replace {SLOT} placeholders with Faker-generated values."""

    def _replace(match: re.Match) -> str:
        slot = match.group(1)
        if slot not in SLOT_REGISTRY:
            raise KeyError(f"Unregistered slot: {slot!r}")
        return SLOT_REGISTRY[slot](faker)

    return _SLOT_RE.sub(_replace, text_template)


# ── Main Generator ───────────────────────────────────────────────────────


def generate_from_templates(
    templates_dir: str | Path,
    count: int,
    seed: int | None = None,
) -> list[RawExample]:
    """Generate synthetic examples by filling YAML templates with Faker data.

    Spec: specifications/data-generation.md §2.1
    """
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")

    domain_files = load_templates(templates_dir)
    faker = Faker()
    if seed is not None:
        Faker.seed(seed)
        faker.seed_instance(seed)

    results: list[RawExample] = []
    num_domains = len(domain_files)

    for i in range(count):
        # Round-robin across domains
        domain_file = domain_files[i % num_domains]
        # Cycle through patterns within the domain
        pattern = domain_file.patterns[i // num_domains % len(domain_file.patterns)]

        text = fill_slots(pattern.text_template, faker)
        rationale = pattern.rationale_template if pattern.rationale_template else ""

        results.append(
            RawExample(
                text=text,
                labels=pattern.labels,
                risk=pattern.risk,
                rationale=rationale,
                is_hard_negative=False,
                domain=domain_file.domain,
                source="template",
            )
        )

    return results

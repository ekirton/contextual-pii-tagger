# Data Generation Pipeline

**Features:** F-02 (Synthetic Dataset Generation), F-06 (Human Review Workflow)

---

## 1. Pipeline Overview

The data generation pipeline produces 20,000 Example records through five sequential stages. Each stage takes the output of the previous stage as input, progressively building and refining the dataset.

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│  Stage 1   │─┬─>│  Stage 2   │───>│  Stage 3   │─┬─>│  Stage 4   │───>│  Stage 5   │
│  Template  │ │  │    LLM     │    │   Auto-    │ │  │    Hard    │    │   Human    │
│ Generation │ │  │ Augmented  │    │  Labeling  │ │  │ Negatives  │    │   Review   │
└────────────┘ │  └────────────┘    └────────────┘ │  └────────────┘    └────────────┘
     │         │        │                 │         │        │                 │
     v         │        v                 v         │        v                 v
  Template-    │  LLM-generated     Validated       │  Hard-negative      Final reviewed
  based Example│  Example records   labels on LLM   │  Example records    Example records
  records      └─────(bypass)──────>examples only───┘  added              (1% corrected)
```

## 2. Stage 1 — Template-Based Generation

**Input:** Domain template files (YAML), Faker library
**Output:** Example records with `source: template`

Parameterized templates define common prompt structures for each domain. The Faker library fills template slots with synthetic but realistic values (names, addresses, organizations, dates).

### Domain Templates

| Template File | Domain | Example Prompt Patterns |
|---------------|--------|------------------------|
| `medical.yaml` | Medical context | Doctor visits, specialist referrals, health-adjacent questions |
| `scheduling.yaml` | Routines | Commute patterns, recurring appointments, school schedules |
| `workplace.yaml` | Workplace | Office locations, departments, job roles, team descriptions |
| `personal.yaml` | Personal | Introductions, neighborhood descriptions, family context |

Each template includes annotations marking which SpanLabel categories are present in the generated text. When a template is filled, the annotations carry through to the Example's label set.

### Volume Target

Template-based generation produces the bulk of the training set. The exact proportion relative to LLM-augmented examples is a tuning parameter, but templates are the primary source for high-volume, predictable coverage of the taxonomy.

## 3. Stage 2 — LLM-Augmented Generation

**Input:** Taxonomy definition, domain descriptions, diversity targets
**Output:** Example records with `source: llm-augmented`

A local LLM served via Ollama generates diverse, naturalistic examples that are difficult to template — particularly quasi-identifier combinations, ambiguous cases, and examples with multiple overlapping categories. The default model is Qwen 2.5 7B Instruct (Q4_K_M quantization), chosen for strong structured JSON output on consumer Apple Silicon hardware.

### Prompt Strategy

The generation prompt provides:
- The full Tier 2 taxonomy with definitions and examples
- The target domain and SpanLabel distribution for the batch
- Instructions to produce both the text and the category label annotations in a structured format

### Diversity Controls

- Each generation batch targets a specific domain and SpanLabel combination to ensure coverage
- Batches are distributed across domains to prevent overrepresentation
- The prompt explicitly requests variation in sentence structure, tone, and context

## 4. Stage 3 — Auto-Labeling Validation

**Input:** LLM-generated Example records from Stage 2 (template-sourced examples from Stage 1 are passed through without validation)
**Output:** Example records with validated Finding lists and RiskLevel assignments

A second LLM pass reviews each LLM-generated Example's annotations:
- Verifies that each category label is correct for the text content
- Checks for missing labels (categories present in the text but not annotated)
- Assigns or validates the RiskLevel based on the combination of labels
- Generates the rationale string for examples with risk MEDIUM or HIGH

Template-sourced examples skip this stage because their labels are defined by the templates themselves and do not require LLM verification.

Examples where the validation pass disagrees with the original annotations are flagged for review or regenerated.

## 5. Stage 4 — Hard Negative Injection

**Input:** Validated Example records from Stage 3
**Output:** Dataset with 10% hard negatives per split

Hard negatives are Example records where `is_hard_negative: true` and `labels` is an empty set. They contain text that mentions places, times, organizations, or other details that resemble quasi-identifiers but are not PII in context.

### Hard Negative Categories

| Category | Examples |
|----------|----------|
| Historical references | "The Battle of Gettysburg took place in July 1863" |
| Fictional characters | "Sherlock Holmes lived at 221B Baker Street" |
| Public figures in public contexts | "The CEO of Apple announced the new product at their Cupertino headquarters" |
| Generic statements | "Many hospitals in the Portland area offer pediatric care" |
| Hypothetical scenarios | "Imagine someone who works at a hospital and commutes every Tuesday" |

### Distribution

Hard negatives are injected proportionally so that each split (train, validation, test) contains exactly 10% hard negatives. They are generated with `source: hard-negative`.

## 6. Stage 5 — Human Review

**Input:** Complete dataset with all splits
**Output:** Corrected dataset

A random 1% sample of Example records across all splits is selected for manual review using Label Studio. Annotators verify:
- Category labels are correct and complete (no missing or extraneous labels)
- RiskLevel is appropriate for the combination of labels
- Hard negatives are truly non-PII in context
- Rationale strings are accurate

Corrections are applied directly to the Example records. The review does not add new examples or change the split assignments.

## 7. Output Format

The final dataset is a single file (or set of split files) containing Example records in JSON Lines format, one Example per line:

```json
{
  "id": "train-00001",
  "text": "I dropped my daughter off at Jefferson Elementary this morning...",
  "labels": ["LOCATION", "ROUTINE"],
  "risk": "MEDIUM",
  "rationale": "School name combined with daily routine narrows location and schedule.",
  "is_hard_negative": false,
  "split": "train",
  "domain": "personal",
  "source": "template"
}
```

## 8. Split Assignment

After all examples are generated, they are shuffled and assigned to splits:
- **Train:** 16,000 examples (80%)
- **Validation:** 2,000 examples (10%)
- **Test:** 2,000 examples (10%)

Split assignment happens before human review so that the review sample is drawn proportionally from all splits.

#!/usr/bin/env bash
#
# Generate synthetic training data using the full generation pipeline.
#
# Usage:
#   ./scripts/generate-training-data.sh [OPTIONS]
#
# Options:
#   -c, --count    Total number of examples to generate (default: 1000)
#   -s, --seed     Random seed for reproducibility (default: none)
#   -o, --output   Output directory (default: data/)
#   -m, --model    Ollama model tag for generation/validation (default: qwen2.5:7b)
#   -t, --template-fraction  Fraction from templates vs LLM (default: 0.5)
#   --templates-only  Skip LLM stages, use only template generation
#   -h, --help     Show this help message
#
# If existing JSONL files are found in the output directory, the pipeline
# resumes by generating only the remaining examples and appending.
#
# Output:
#   {output}/train.jsonl
#   {output}/validation.jsonl
#   {output}/test.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATES_DIR="$PROJECT_ROOT/src/contextual_pii_tagger/data/templates"

COUNT=1000
SEED=""
OUTPUT_DIR="$PROJECT_ROOT/data"
MODEL="qwen2.5:7b"
TEMPLATE_FRACTION="0.5"
TEMPLATES_ONLY=""

usage() {
    sed -n '3,20p' "$0" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--count)    COUNT="$2";             shift 2 ;;
        -s|--seed)     SEED="$2";              shift 2 ;;
        -o|--output)   OUTPUT_DIR="$2";        shift 2 ;;
        -m|--model)    MODEL="$2";             shift 2 ;;
        -t|--template-fraction) TEMPLATE_FRACTION="$2"; shift 2 ;;
        --templates-only) TEMPLATES_ONLY="1";  shift ;;
        -h|--help)     usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

exec python -c "
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(message)s')

count = int('${COUNT}')
seed_str = '${SEED}'
seed = int(seed_str) if seed_str else None
output_dir = '${OUTPUT_DIR}'
templates_dir = '${TEMPLATES_DIR}'
model = '${MODEL}'
template_fraction = float('${TEMPLATE_FRACTION}')
templates_only = bool('${TEMPLATES_ONLY}')

if templates_only:
    # Fast path: templates only, no LLM calls
    from contextual_pii_tagger.data.templates import generate_from_templates
    from contextual_pii_tagger.data.generate import assign_splits_and_ids
    from contextual_pii_tagger.data.dataset_io import append_dataset, dataset_stats, write_dataset

    existing = dataset_stats(output_dir)
    existing_count = existing.total if existing else 0
    remaining = max(0, count - existing_count)

    if remaining == 0:
        print(f'Target of {count} already reached ({existing_count} existing). Nothing to generate.')
        sys.exit(0)

    if existing:
        print(f'Found {existing_count} existing examples. Generating {remaining} more...')

    raw = generate_from_templates(templates_dir, remaining, seed=seed)
    id_offsets = existing.max_id_by_split if existing else {}
    examples = assign_splits_and_ids(raw, seed=seed, id_offset_by_split=id_offsets)

    if existing:
        append_dataset(examples, output_dir)
    else:
        write_dataset(examples, output_dir)

    final = dataset_stats(output_dir)
    for split, n in sorted(final.by_split.items()):
        print(f'  {split}: {n} examples')
    print(f'Done. {final.total} total examples in {output_dir}/')
else:
    # Full pipeline with LLM stages
    from contextual_pii_tagger.data.generate import GenerationConfig, generate_dataset

    config = GenerationConfig(
        templates_dir=templates_dir,
        total_count=count,
        template_fraction=template_fraction,
        seed=seed,
        model=model,
        output_dir=output_dir,
    )
    dataset = generate_dataset(config)

    splits = {}
    for ex in dataset:
        splits[ex.split] = splits.get(ex.split, 0) + 1
    for split, n in sorted(splits.items()):
        print(f'  {split}: {n} examples')

    hn = sum(1 for ex in dataset if ex.is_hard_negative)
    print(f'Done. {len(dataset)} examples ({hn} hard negatives) written to {output_dir}/')
"

#!/usr/bin/env bash
#
# Compare the fine-tuned LoRA model against the XGBoost baseline.
#
# Usage:
#   ./scripts/compare-lora-vs-xgboost.sh [OPTIONS]
#
# Options:
#   -d, --data-dir       Data directory (default: data/)
#   -m, --model-path     Path to merged or adapter model (default: output/merged-model)
#   -b, --baseline-path  Path to saved XGBoost baseline (default: output/xgboost-baseline)
#   -h, --help           Show this help message
#
# Output:
#   Side-by-side metrics with deltas (fine-tuned minus baseline).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="$PROJECT_ROOT/data"
MODEL_PATH="$PROJECT_ROOT/output/merged-model"
BASELINE_PATH="$PROJECT_ROOT/output/xgboost-baseline"

usage() {
    sed -n '3,15p' "$0" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--data-dir)      DATA_DIR="$2";      shift 2 ;;
        -m|--model-path)    MODEL_PATH="$2";     shift 2 ;;
        -b|--baseline-path) BASELINE_PATH="$2";  shift 2 ;;
        -h|--help)          usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

TEST_FILE="$DATA_DIR/test.jsonl"
if [[ ! -f "$TEST_FILE" ]]; then
    echo "Error: Test data not found at $TEST_FILE" >&2
    echo "Run ./scripts/generate-training-data.sh first." >&2
    exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: Model not found at $MODEL_PATH" >&2
    echo "Run ./scripts/train-lora.sh first." >&2
    exit 1
fi

if [[ ! -f "$BASELINE_PATH/xgboost_baseline.pkl" ]]; then
    echo "Error: XGBoost baseline not found at $BASELINE_PATH" >&2
    echo "Run ./scripts/train-xgboost.sh first." >&2
    exit 1
fi

echo "==> Comparing fine-tuned LoRA vs XGBoost baseline"
echo "    Model: $MODEL_PATH"
echo "    Baseline: $BASELINE_PATH"
echo "    Data: $DATA_DIR"

python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

from contextual_pii_tagger.detector import PIIDetector
from contextual_pii_tagger.data.dataset_io import read_dataset
from contextual_pii_tagger.eval.baseline import XGBoostPredictor
from contextual_pii_tagger.eval.evaluate import evaluate, compare_models

all_data = read_dataset('${DATA_DIR}')
test_data = [ex for ex in all_data if ex.split == 'test']

# ── Fine-tuned model ─────────────────────────────────────────────
print(f'Loading fine-tuned model from ${MODEL_PATH}...')
detector = PIIDetector.from_pretrained('${MODEL_PATH}')
print(f'Evaluating fine-tuned model on {len(test_data)} test examples...')
ft_report = evaluate(detector, test_data, model_name='fine-tuned')

# ── XGBoost baseline ─────────────────────────────────────────────
print(f'Loading XGBoost baseline from ${BASELINE_PATH}...')
baseline = XGBoostPredictor.load('${BASELINE_PATH}')
print(f'Evaluating XGBoost baseline on {len(test_data)} test examples...')
bl_report = evaluate(baseline, test_data, model_name='XGBoost')

# ── Comparison ────────────────────────────────────────────────────
comparison = compare_models(ft_report, bl_report)

print()
print(f'=== Model Comparison ({ft_report.test_set_size} test examples) ===')
print()
print(f'{\"Metric\":<28s} {\"LoRA\":>8s} {\"XGBoost\":>8s} {\"Delta\":>8s}')
print('-' * 56)

metrics = [
    ('Multilabel F1',          'multilabel_f1'),
    ('Risk accuracy',          'risk_accuracy'),
    ('False negative rate',    'false_negative_rate'),
    ('QUASI-ID F1',            'quasi_id_f1'),
    ('Hard negative precision', 'hard_negative_precision'),
]

for display_name, key in metrics:
    ft_val = comparison['finetuned'][key]
    bl_val = comparison['baseline'][key]
    delta = comparison['deltas'][key]
    sign = '+' if delta >= 0 else ''
    print(f'{display_name:<28s} {ft_val:>8.4f} {bl_val:>8.4f} {sign}{delta:>7.4f}')

print()
print('Per-label F1:')
print(f'{\"Label\":<20s} {\"LoRA\":>8s} {\"XGBoost\":>8s} {\"Delta\":>8s}')
print('-' * 48)
for label_name in sorted(comparison['f1_by_label'].keys()):
    entry = comparison['f1_by_label'][label_name]
    delta = entry['delta']
    sign = '+' if delta >= 0 else ''
    print(f'{label_name:<20s} {entry[\"finetuned\"]:>8.4f} {entry[\"baseline\"]:>8.4f} {sign}{delta:>7.4f}')

print()
print('Done.')
"

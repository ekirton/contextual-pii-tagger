#!/usr/bin/env bash
#
# Train the XGBoost baseline classifier for comparison against the fine-tuned model.
#
# Usage:
#   ./scripts/train-xgboost.sh [OPTIONS]
#
# Options:
#   -d, --data-dir   Training data directory (default: data/)
#   -o, --output-dir Output directory (default: output/xgboost-baseline)
#   -h, --help       Show this help message
#
# Output:
#   {output-dir}/xgboost_baseline.pkl   Trained XGBoost model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="$PROJECT_ROOT/data"
OUTPUT_DIR="$PROJECT_ROOT/output/xgboost-baseline"

usage() {
    sed -n '3,14p' "$0" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--data-dir)   DATA_DIR="$2";    shift 2 ;;
        -o|--output-dir) OUTPUT_DIR="$2";   shift 2 ;;
        -h|--help)       usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

TRAIN_FILE="$DATA_DIR/train.jsonl"
if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "Error: Training data not found at $TRAIN_FILE" >&2
    echo "Run ./scripts/generate-training-data.sh first." >&2
    exit 1
fi

echo "==> Training XGBoost baseline"
echo "    Data: $DATA_DIR"
echo "    Output: $OUTPUT_DIR"

python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

from contextual_pii_tagger.data.dataset_io import read_dataset
from contextual_pii_tagger.eval.baseline import train_baseline
from contextual_pii_tagger.eval.evaluate import evaluate

all_data = read_dataset('${DATA_DIR}')
train_data = [ex for ex in all_data if ex.split == 'train']
test_data = [ex for ex in all_data if ex.split == 'test']

print(f'Training on {len(train_data)} examples...')
baseline = train_baseline(train_data)
baseline.save('${OUTPUT_DIR}')

print(f'Evaluating on {len(test_data)} test examples...')
report = evaluate(baseline, test_data, model_name='XGBoost')

print()
print('=== XGBoost Baseline Results ===')
print(f'  Multilabel F1:          {report.multilabel_f1:.4f}')
print(f'  Risk accuracy:          {report.risk_accuracy:.4f}')
print(f'  False negative rate:    {report.false_negative_rate:.4f}')
print(f'  QUASI-ID F1:            {report.quasi_id_f1:.4f}')
print(f'  Hard negative precision:{report.hard_negative_precision:.4f}')
print()
print('Per-label F1:')
for label, f1 in sorted(report.f1_by_label.items(), key=lambda x: x[0].value):
    print(f'  {label.value:<20s} {f1:.4f}')
print()
print('Done.')
"

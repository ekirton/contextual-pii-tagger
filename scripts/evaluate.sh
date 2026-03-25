#!/usr/bin/env bash
#
# Evaluate a fine-tuned model on the test split.
#
# Usage:
#   ./scripts/evaluate.sh [OPTIONS]
#
# Options:
#   -d, --data-dir    Data directory containing test.jsonl (default: data/)
#   -m, --model-path  Path to merged or adapter model (default: output/merged-model)
#   -n, --model-name  Display name for the report (default: fine-tuned)
#   -h, --help        Show this help message
#
# Output:
#   Prints evaluation metrics against the success criteria from DEVELOPMENT.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="$PROJECT_ROOT/data"
MODEL_PATH="$PROJECT_ROOT/output/merged-model"
MODEL_NAME="fine-tuned"

usage() {
    sed -n '3,14p' "$0" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--data-dir)   DATA_DIR="$2";   shift 2 ;;
        -m|--model-path) MODEL_PATH="$2"; shift 2 ;;
        -n|--model-name) MODEL_NAME="$2"; shift 2 ;;
        -h|--help)       usage ;;
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

echo "==> Evaluating model: $MODEL_NAME"
echo "    Model: $MODEL_PATH"
echo "    Data: $DATA_DIR"

python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

from contextual_pii_tagger.detector import PIIDetector
from contextual_pii_tagger.data.dataset_io import read_dataset
from contextual_pii_tagger.eval.evaluate import evaluate

all_data = read_dataset('${DATA_DIR}')
test_data = [ex for ex in all_data if ex.split == 'test']
print(f'Loaded {len(test_data)} test examples')

detector = PIIDetector.from_pretrained('${MODEL_PATH}')
report = evaluate(detector, test_data, model_name='${MODEL_NAME}')

# Targets from DEVELOPMENT.md
targets = {
    'multilabel_f1':          0.80,
    'risk_accuracy':          0.85,
    'false_negative_rate':    0.08,  # upper bound
    'quasi_id_f1':            0.70,
    'hard_negative_precision': 0.92,
}

print()
print(f'=== Evaluation: {report.model_name} ({report.test_set_size} examples) ===')
print()
print(f'{\"Metric\":<28s} {\"Value\":>8s} {\"Target\":>8s} {\"Pass\":>6s}')
print('-' * 54)

def check(name, value, target, lower_is_better=False):
    if lower_is_better:
        passed = value <= target
        op = '<='
    else:
        passed = value >= target
        op = '>='
    status = 'YES' if passed else 'NO'
    print(f'{name:<28s} {value:>8.4f} {op} {target:<5.2f} {status:>4s}')
    return passed

results = []
results.append(check('Multilabel F1',          report.multilabel_f1,          targets['multilabel_f1']))
results.append(check('Risk accuracy',          report.risk_accuracy,          targets['risk_accuracy']))
results.append(check('False negative rate',    report.false_negative_rate,    targets['false_negative_rate'], lower_is_better=True))
results.append(check('QUASI-ID F1',            report.quasi_id_f1,            targets['quasi_id_f1']))
results.append(check('Hard negative precision', report.hard_negative_precision, targets['hard_negative_precision']))

print()
print('Per-label F1:')
for label, f1 in sorted(report.f1_by_label.items(), key=lambda x: x[0].value):
    print(f'  {label.value:<20s} {f1:.4f}')

print()
all_pass = all(results)
if all_pass:
    print('All targets met.')
else:
    print(f'{sum(results)}/{len(results)} targets met.')
"

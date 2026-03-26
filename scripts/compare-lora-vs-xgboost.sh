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
#   Writes comparison report to <data-dir>/comparison-report.txt.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="$PROJECT_ROOT/data"
MODEL_PATH="$PROJECT_ROOT/output/merged-model"
BASELINE_PATH="$PROJECT_ROOT/output/xgboost-baseline"
usage() {
    sed -n '3,16p' "$0" | sed 's/^# \?//'
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

REPORT_FILE="$DATA_DIR/comparison-report.txt"

echo "==> Comparing fine-tuned LoRA vs XGBoost baseline"
echo "    Model: $MODEL_PATH"
echo "    Baseline: $BASELINE_PATH"
echo "    Data: $DATA_DIR"
echo "    Report: $REPORT_FILE"

# Run each model in a separate Python process to avoid libomp conflicts
# between PyTorch and XGBoost on macOS ARM.
FT_REPORT_FILE=$(mktemp "${TMPDIR:-/tmp}/ft-report-XXXXXX.json")
trap 'rm -f "$FT_REPORT_FILE"' EXIT

# ── Step 1: Evaluate fine-tuned model (loads PyTorch) ─────────────
echo "--- Evaluating fine-tuned model ---"
python -c "
import json, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

from contextual_pii_tagger.detector import PIIDetector
from contextual_pii_tagger.data.dataset_io import read_dataset
from contextual_pii_tagger.eval.evaluate import evaluate

all_data = read_dataset('${DATA_DIR}')
test_data = [ex for ex in all_data if ex.split == 'test']

print(f'Loading fine-tuned model from ${MODEL_PATH}...')
detector = PIIDetector.from_pretrained('${MODEL_PATH}')
print(f'Evaluating fine-tuned model on {len(test_data)} test examples...')
ft_report = evaluate(detector, test_data, model_name='fine-tuned')

# Serialize report to JSON for the comparison step
report_dict = {
    'model_name': ft_report.model_name,
    'test_set_size': ft_report.test_set_size,
    'multilabel_f1': ft_report.multilabel_f1,
    'f1_by_label': {l.value: v for l, v in ft_report.f1_by_label.items()},
    'risk_accuracy': ft_report.risk_accuracy,
    'false_negative_rate': ft_report.false_negative_rate,
    'quasi_id_f1': ft_report.quasi_id_f1,
    'hard_negative_precision': ft_report.hard_negative_precision,
}
with open('${FT_REPORT_FILE}', 'w') as f:
    json.dump(report_dict, f)
print('Fine-tuned evaluation complete.')
"

# ── Step 2: Evaluate XGBoost baseline and compare (loads XGBoost) ──
echo "--- Evaluating XGBoost baseline ---"
python -c "
import json, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

from contextual_pii_tagger.data.dataset_io import read_dataset
from contextual_pii_tagger.entities import SpanLabel
from contextual_pii_tagger.eval.baseline import XGBoostPredictor
from contextual_pii_tagger.eval.evaluate import evaluate, compare_models
from contextual_pii_tagger.example import EvaluationReport

all_data = read_dataset('${DATA_DIR}')
test_data = [ex for ex in all_data if ex.split == 'test']

print(f'Loading XGBoost baseline from ${BASELINE_PATH}...')
baseline = XGBoostPredictor.load('${BASELINE_PATH}')
print(f'Evaluating XGBoost baseline on {len(test_data)} test examples...')
bl_report = evaluate(baseline, test_data, model_name='XGBoost')

# Reload fine-tuned report from JSON
with open('${FT_REPORT_FILE}') as f:
    ft_dict = json.load(f)

label_lookup = {l.value: l for l in SpanLabel}
ft_report = EvaluationReport(
    model_name=ft_dict['model_name'],
    test_set_size=ft_dict['test_set_size'],
    multilabel_f1=ft_dict['multilabel_f1'],
    f1_by_label={label_lookup[k]: v for k, v in ft_dict['f1_by_label'].items()},
    risk_accuracy=ft_dict['risk_accuracy'],
    false_negative_rate=ft_dict['false_negative_rate'],
    quasi_id_f1=ft_dict['quasi_id_f1'],
    hard_negative_precision=ft_dict['hard_negative_precision'],
)

# ── Comparison ────────────────────────────────────────────────────
report_path = '${REPORT_FILE}'
print('Writing comparison report...')
comparison = compare_models(ft_report, bl_report)

with open(report_path, 'w') as f:
    f.write(f'=== Model Comparison ({ft_report.test_set_size} test examples) ===\n')
    f.write('\n')
    f.write(f'{\"Metric\":<28s} {\"LoRA\":>8s} {\"XGBoost\":>8s} {\"Delta\":>8s}\n')
    f.write('-' * 56 + '\n')

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
        f.write(f'{display_name:<28s} {ft_val:>8.4f} {bl_val:>8.4f} {sign}{delta:>7.4f}\n')

    f.write('\n')
    f.write('Per-label F1:\n')
    f.write(f'{\"Label\":<20s} {\"LoRA\":>8s} {\"XGBoost\":>8s} {\"Delta\":>8s}\n')
    f.write('-' * 48 + '\n')
    for label_name in sorted(comparison['f1_by_label'].keys()):
        entry = comparison['f1_by_label'][label_name]
        delta = entry['delta']
        sign = '+' if delta >= 0 else ''
        f.write(f'{label_name:<20s} {entry[\"finetuned\"]:>8.4f} {entry[\"baseline\"]:>8.4f} {sign}{delta:>7.4f}\n')

print(f'Report written to {report_path}')
"

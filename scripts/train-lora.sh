#!/usr/bin/env bash
#
# Fine-tune, merge, and optionally convert the contextual PII tagger model.
#
# Usage:
#   ./scripts/train.sh [OPTIONS]
#
# Options:
#   -d, --data-dir       Training data directory (default: data/)
#   -c, --config         Training config YAML (default: src/.../train/config.yaml)
#   -o, --output-dir     Adapter output directory (default: output/contextual-pii-tagger)
#   -m, --merged-dir     Merged model output directory (default: output/merged-model)
#   --skip-merge         Skip adapter merge step
#   --gguf               Convert merged model to GGUF after merge
#   --gguf-output        GGUF output path (default: ~/.cache/contextual-pii-tagger/model.gguf)
#   --gguf-type          GGUF quantization type (default: q4_k_m)
#   --llama-cpp-dir      Path to llama.cpp repo (default: llama.cpp)
#   -h, --help           Show this help message
#
# Output:
#   {output-dir}/    LoRA adapter weights
#   {merged-dir}/    Standalone merged model (unless --skip-merge)
#   {gguf-output}    GGUF file (only with --gguf)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="$PROJECT_ROOT/data"
CONFIG="$PROJECT_ROOT/src/contextual_pii_tagger/train/config.yaml"
OUTPUT_DIR="$PROJECT_ROOT/output/contextual-pii-tagger"
MERGED_DIR="$PROJECT_ROOT/output/merged-model"
SKIP_MERGE=""
GGUF=""
GGUF_OUTPUT="$HOME/.cache/contextual-pii-tagger/model.gguf"
GGUF_TYPE="q4_k_m"
LLAMA_CPP_DIR="$PROJECT_ROOT/llama.cpp"

usage() {
    sed -n '3,24p' "$0" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--data-dir)      DATA_DIR="$2";        shift 2 ;;
        -c|--config)        CONFIG="$2";           shift 2 ;;
        -o|--output-dir)    OUTPUT_DIR="$2";       shift 2 ;;
        -m|--merged-dir)    MERGED_DIR="$2";       shift 2 ;;
        --skip-merge)       SKIP_MERGE="1";        shift ;;
        --gguf)             GGUF="1";              shift ;;
        --gguf-output)      GGUF_OUTPUT="$2";      shift 2 ;;
        --gguf-type)        GGUF_TYPE="$2";        shift 2 ;;
        --llama-cpp-dir)    LLAMA_CPP_DIR="$2";    shift 2 ;;
        -h|--help)          usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Validate inputs ──────────────────────────────────────────────────
TRAIN_FILE="$DATA_DIR/train.jsonl"
if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "Error: Training data not found at $TRAIN_FILE" >&2
    echo "Run ./scripts/generate-training-data.sh first." >&2
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config not found at $CONFIG" >&2
    exit 1
fi

# ── Step 1: Fine-tune ────────────────────────────────────────────────
echo "==> Fine-tuning with config: $CONFIG"
echo "    Data: $DATA_DIR"
echo "    Output: $OUTPUT_DIR"

python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

from contextual_pii_tagger.train.train import load_config, train
from contextual_pii_tagger.train.data_utils import prepare_dataset
from transformers import AutoTokenizer

config = load_config('${CONFIG}')
config['output_dir'] = '${OUTPUT_DIR}'

tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
dataset = prepare_dataset('${DATA_DIR}', tokenizer)
print(f'Prepared {len(dataset)} training examples')

train(config, dataset)
print('Training complete.')
"

# ── Step 2: Merge adapter ────────────────────────────────────────────
if [[ -z "$SKIP_MERGE" ]]; then
    echo ""
    echo "==> Merging adapter into base model"
    echo "    Adapter: $OUTPUT_DIR"
    echo "    Output: $MERGED_DIR"

    python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

from contextual_pii_tagger.train.train import load_config
from contextual_pii_tagger.train.merge import merge_adapter

config = load_config('${CONFIG}')

merge_adapter(
    base_model_path=config['base_model'],
    adapter_path='${OUTPUT_DIR}',
    output_path='${MERGED_DIR}',
)
print('Merge complete.')
"
fi

# ── Step 3: Convert to GGUF (optional) ───────────────────────────────
if [[ -n "$GGUF" ]]; then
    if [[ -z "$SKIP_MERGE" || -d "$MERGED_DIR" ]]; then
        echo ""
        echo "==> Converting to GGUF"
        echo "    Input: $MERGED_DIR"
        echo "    Output: $GGUF_OUTPUT"
        echo "    Quantization: $GGUF_TYPE"

        CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
        if [[ ! -f "$CONVERT_SCRIPT" ]]; then
            echo "Error: llama.cpp not found at $LLAMA_CPP_DIR" >&2
            echo "Clone it: git clone https://github.com/ggerganov/llama.cpp.git" >&2
            exit 1
        fi

        mkdir -p "$(dirname "$GGUF_OUTPUT")"

        python "$CONVERT_SCRIPT" \
            "$MERGED_DIR" \
            --outfile "$GGUF_OUTPUT" \
            --outtype "$GGUF_TYPE"

        echo "GGUF conversion complete: $GGUF_OUTPUT"
    else
        echo "Error: --gguf requires a merged model. Remove --skip-merge or ensure $MERGED_DIR exists." >&2
        exit 1
    fi
fi

echo ""
echo "Done."
